import sys
sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# v12 — SOM PUREZA POR FEATURE + CNN LOBO
#
# Seleccion de features (NO supervisada):
#   Para cada feature individualmente se entrena un SOM 5x5 pequeño.
#   Se mide la pureza resultante (fraccion de la clase mayoritaria en cada
#   neurona activa). Las 30 features con mayor pureza media se pasan a la CNN.
#   El SOM nunca ve las etiquetas durante el entrenamiento.
#   Las etiquetas solo se usan al final para calcular la pureza (evaluacion).
#
# CNN: esquema LOBO v9 — 5 clases, 5 folds
# =============================================================================

# ── Configuracion CNN (identica a v9) ─────────────────────────────────────────
N_CLASES = 5
NOMBRES_CLASE = {0:'Sano', 1:'OR-Grabado', 2:'OR-Taladro',
                 3:'OR-Desgaste', 4:'IR'}
EXCLUIDOS  = {'KA01', 'KI01'}
ETIQUETA_ROD = {
    'K001':0,'K002':0,'K003':0,'K004':0,'K005':0,'K006':0,
    'KA03':1,'KA05':1,'KA06':1,
    'KA07':2,'KA08':2,'KA09':2,
    'KA04':3,'KA15':3,'KA16':3,'KA22':3,'KA30':3,
    'KI03':4,'KI04':4,'KI05':4,'KI07':4,'KI08':4,
    'KI14':4,'KI16':4,'KI17':4,'KI18':4,'KI21':4,
}
LOBO_TEST  = ['K003','KA06','KA09','KA22','KI07']
TRAIN_FIJO = {
    'Sano':       ['K001','K002','K004','K005','K006'],
    'OR-Grabado': ['KA03','KA05'],
    'OR-Taladro': ['KA07','KA08'],
    'OR-Desgaste':['KA04','KA15','KA16','KA30'],
    'IR':         ['KI03','KI04','KI05','KI08',
                   'KI14','KI16','KI17','KI18','KI21'],
}
TODOS_RODS = LOBO_TEST + [r for rods in TRAIN_FIJO.values() for r in rods]

RUTA_X     = r"\X_feat_crudos.npy"
RUTA_COD   = r"\Y_codigos_crudos.npy"
RUTA_NAMES = r"\feature_names_crudos.npy"

# ── Parametros SOM ────────────────────────────────────────────────────────────
SOM_ROWS    = 5          # grilla pequena por feature
SOM_COLS    = 5
SOM_ITER    = 20000      # iteraciones por feature
SOM_LR_INI  = 0.5
SOM_LR_FIN  = 0.05
SOM_SIG_INI = 2.5
SOM_SIG_FIN = 0.5
SOM_SAMPLE  = 10000      # submuestreo para entrenar cada SOM
TOP_FEAT    = 30         # features finales para la CNN
SEED        = 42

# ── Parametros CNN ────────────────────────────────────────────────────────────
EPOCAS     = 60
BATCH_SIZE = 256
LR         = 0.001
EXPERIMENTO = "CNN Top-30 SOM-Pureza | LOBO 5-folds | 5 clases"


# =============================================================================
# SOM MINIMO (1 feature)
# =============================================================================
class MiniSOM:
    def __init__(self, rows, cols, seed=42):
        self.rows = rows; self.cols = cols
        self.n_neu = rows * cols
        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal((self.n_neu, 1)).astype(np.float32)
        rc = np.array([[r,c] for r in range(rows) for c in range(cols)],
                      dtype=np.float32)
        d  = rc[:,None,:] - rc[None,:,:]
        self._d2 = (d**2).sum(axis=2)

    def entrenar(self, X1d, n_iter, lr_ini, lr_fin, sig_ini, sig_fin):
        N   = len(X1d)
        rng = np.random.default_rng(SEED)
        xmn, xmx = X1d.min(), X1d.max()
        self.W = (rng.random((self.n_neu,1)).astype(np.float32)
                  * (xmx - xmn) + xmn)
        for it in range(n_iter):
            x   = X1d[rng.integers(0, N)]
            alp = lr_ini  * (lr_fin  / lr_ini)  ** (it / n_iter)
            sig = sig_ini * (sig_fin / sig_ini) ** (it / n_iter)
            bmu = int(np.argmin((self.W[:,0] - x)**2))
            h   = np.exp(-self._d2[bmu] / (2*sig**2))[:,None].astype(np.float32)
            self.W += alp * h * (x - self.W)

    def mapear(self, X1d):
        return np.array([int(np.argmin((self.W[:,0]-x)**2)) for x in X1d])


def pureza_som(bmus, Y_cls, n_neu, n_clases):
    """Pureza media de las neuronas activas."""
    votos = np.zeros((n_neu, n_clases), dtype=int)
    for idx, cls in zip(bmus, Y_cls):
        votos[idx, cls] += 1
    hits = votos.sum(axis=1)
    activas = hits > 0
    if not activas.any():
        return 0.0
    pur = votos[activas, votos[activas].argmax(axis=1)] / hits[activas]
    return float(pur.mean())


# =============================================================================
# SELECCION DE FEATURES POR PUREZA SOM
# =============================================================================
def seleccionar_features_som(X_clean, Y_cls, names, top_k=TOP_FEAT):
    """
    Para cada feature entrena un MiniSOM 5x5 y mide la pureza.
    Devuelve los indices de las top_k features mas separables.
    """
    n_feat   = X_clean.shape[1]
    purezas  = np.zeros(n_feat)
    rng      = np.random.default_rng(SEED)

    # Submuestreo fijo para todos los SOMs
    N = len(X_clean)
    idx_s = rng.choice(N, min(SOM_SAMPLE, N), replace=False)
    X_sub = X_clean[idx_s]
    Y_sub = Y_cls[idx_s]

    print(f"\n  Evaluando {n_feat} features con MiniSOM {SOM_ROWS}x{SOM_COLS}...")
    for fi in tqdm(range(n_feat), desc="  SOM por feature"):
        x1d = X_sub[:, fi].astype(np.float32)
        som = MiniSOM(SOM_ROWS, SOM_COLS, seed=SEED)
        som.entrenar(x1d, SOM_ITER, SOM_LR_INI, SOM_LR_FIN,
                     SOM_SIG_INI, SOM_SIG_FIN)
        bmus = som.mapear(x1d)
        purezas[fi] = pureza_som(bmus, Y_sub, SOM_ROWS*SOM_COLS, N_CLASES)

    top_idx = np.argsort(purezas)[::-1][:top_k]
    print(f"\n  Top-{top_k} features por pureza SOM:")
    for i, fi in enumerate(top_idx):
        print(f"    {i+1:2d}. {names[fi]:35s} pureza={purezas[fi]:.4f}")

    return top_idx, purezas


# =============================================================================
# MODELO CNN
# =============================================================================
class CNN1D_Multiclass(nn.Module):
    def __init__(self, n_features=30, num_classes=5, dropout=0.4):
        super().__init__()
        self.conv1 = nn.Conv1d(1,  64, kernel_size=5, padding=2)
        self.bn1   = nn.BatchNorm1d(64);  self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64,128, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(128); self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(128,256, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm1d(256)
        self.gap   = nn.AdaptiveAvgPool1d(1)
        self.fc1   = nn.Linear(256, 128); self.drop1 = nn.Dropout(dropout)
        self.fc2   = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x))); x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x))); x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x))); x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.drop1(F.relu(self.fc1(x)))
        return self.fc2(x)


# =============================================================================
# UTILIDADES LOBO
# =============================================================================
def limpiar(X, names):
    X = np.nan_to_num(X, nan=0., posinf=0., neginf=0.)
    m = np.var(X, axis=0) > 1e-15
    return X[:,m], [n for n,k in zip(names,m) if k]

def calcular_fpr(y_true, y_pred, n_clases=N_CLASES):
    fprs = []
    for c in range(n_clases):
        yt = (np.array(y_true)==c).astype(int)
        yp = (np.array(y_pred)==c).astype(int)
        fp = np.sum((yp==1)&(yt==0)); tn = np.sum((yp==0)&(yt==0))
        fprs.append(fp/(fp+tn) if (fp+tn)>0 else 0.)
    return fprs

def construir_folds(rods_presentes):
    todos = [r for r in TODOS_RODS if r in rods_presentes]
    folds = []
    for i, rod_test in enumerate(LOBO_TEST):
        if rod_test not in rods_presentes: continue
        folds.append({'fold_id':i+1,'test':rod_test,
                      'train':[r for r in todos if r!=rod_test],
                      'clase_test':ETIQUETA_ROD[rod_test]})
        print(f"  Fold {i+1}: test={rod_test} "
              f"({NOMBRES_CLASE[ETIQUETA_ROD[rod_test]]})")
    return folds


# =============================================================================
# GRAFICAS
# =============================================================================
def graficar_tabla(mg, nombre_exp, n_folds, guardar='tabla_resultados_v12.png'):
    BG_H='#1a3a5c'; BG_E='#1e2d40'; BR1='#1e2d40'; BR2='#243448'
    FH='#00d4ff';   FW='#ffffff';   FL='#cce8ff'
    cols=['Experimento','Clase','Accuracy','Precision',
          'Sensibilidad','F1-Score','T.Error','FPR']
    cw=[0.16,0.13,0.09,0.10,0.12,0.10,0.10,0.10]
    fig,ax=plt.subplots(figsize=(15, 0.52*(1+N_CLASES)+0.6))
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
    fig.patch.set_facecolor('#0d1b2a')
    xs=np.cumsum([0]+cw[:-1]); rh=1./(N_CLASES+1.5)
    def dc(x,y,w,h,t,bg,fg,fs=9,b=False):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x+.003,y-h+.008),w-.006,h-.01,
            boxstyle="round,pad=0.005",linewidth=0,facecolor=bg,zorder=1))
        ax.text(x+w/2,y-h/2,t,ha='center',va='center',
                fontsize=fs,color=fg,fontweight='bold' if b else 'normal',zorder=2)
    y0=1.
    for j,(c,w) in enumerate(zip(cols,cw)):
        dc(xs[j],y0,w,rh,c,BG_H,FH,9.5,True)
    for ci in range(N_CLASES):
        yr=y0-(ci+1)*rh; bg=BR1 if ci%2==0 else BR2
        if ci==0:
            dc(xs[0],y0-rh,cw[0],rh*N_CLASES,
               f"{nombre_exp}\n({n_folds} folds)",BG_E,FL,7.5)
            dc(xs[2],y0-rh,cw[2],rh*N_CLASES,
               f"{mg['acc_global']:.3f}",BG_E,FW,9)
            dc(xs[5],y0-rh,cw[5],rh*N_CLASES,
               f"{mg['f1_macro']:.3f}",BG_E,FW,9)
        dc(xs[1],yr,cw[1],rh,NOMBRES_CLASE[ci],bg,FW,8)
        dc(xs[3],yr,cw[3],rh,f"{mg['prec'][ci]:.3f}",bg,FW,9)
        dc(xs[4],yr,cw[4],rh,f"{mg['rec'][ci]:.3f}",bg,FW,9)
        dc(xs[6],yr,cw[6],rh,f"{1-mg['rec'][ci]:.3f}",bg,FW,9)
        dc(xs[7],yr,cw[7],rh,f"{mg['fpr'][ci]:.3f}",bg,FW,9)
    plt.tight_layout(pad=0.1)
    plt.savefig(guardar,dpi=180,bbox_inches='tight',facecolor='#0d1b2a')
    plt.show(); print(f"  Tabla: {guardar}")

def graficar_barras_pureza(purezas, names, top_idx,
                            guardar='v12_pureza_features.png'):
    top30_pur = purezas[top_idx]
    top30_nom = [names[i][:22] for i in top_idx]
    colores   = plt.cm.viridis(np.linspace(0.2, 0.9, TOP_FEAT))

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor('#0d1b2a'); ax.set_facecolor('#0d1b2a')
    bars = ax.barh(range(TOP_FEAT), top30_pur[::-1],
                   color=colores[::-1], edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(TOP_FEAT))
    ax.set_yticklabels(top30_nom[::-1], fontsize=8, color='white')
    ax.set_xlabel('Pureza SOM media', color='white', fontsize=11)
    ax.set_title(
        f'Top-{TOP_FEAT} features por Pureza SOM (50x50)\n'
        'Seleccion no supervisada — mayor pureza = mejor separabilidad',
        color='white', fontsize=11, fontweight='bold')
    ax.tick_params(colors='white')
    ax.grid(axis='x', ls='--', alpha=0.3, color='white')
    for spine in ax.spines.values(): spine.set_edgecolor('#333355')
    for bar, val in zip(bars, top30_pur[::-1]):
        ax.text(val+0.002, bar.get_y()+bar.get_height()/2,
                f'{val:.3f}', va='center', color='white', fontsize=7.5)
    plt.tight_layout()
    plt.savefig(guardar, dpi=150, bbox_inches='tight', facecolor='#0d1b2a')
    plt.show(); print(f"  Pureza features: {guardar}")

def graficar_accuracy_barras(resultados_test, etiquetas_por_eval,
                              guardar='v12_lobo_acc.png'):
    color_clase={0:'#42A5F5',1:'#66BB6A',2:'#26C6DA',3:'#FFA726',4:'#EF5350'}
    colores_bar=[color_clase[ETIQUETA_ROD[r]] for r in etiquetas_por_eval]
    fig,ax=plt.subplots(figsize=(max(10,len(etiquetas_por_eval)*2),5))
    fig.patch.set_facecolor('#0d1b2a'); ax.set_facecolor('#0d1b2a')
    ax.bar(etiquetas_por_eval, resultados_test, color=colores_bar,
           edgecolor='white', linewidth=0.8, zorder=3)
    ax.axhline(np.mean(resultados_test), color='white', ls='--', lw=1.8,
               label=f'Media LOBO {np.mean(resultados_test):.1f}%', zorder=4)
    ax.axhline(100/N_CLASES, color='#FF6B6B', ls=':', lw=1.5,
               label=f'Azar ({100/N_CLASES:.1f}%)', zorder=4)
    ax.set_ylim(0,115)
    ax.set_ylabel('Accuracy LOBO (%)', color='white', fontsize=11)
    ax.set_xlabel('Rodamiento evaluado', color='white', fontsize=11)
    ax.set_title(
        f'v12 CNN Top-{TOP_FEAT} (SOM-Pureza) — Accuracy por rodamiento\n'
        'Azul=Sano | Verde=OR-Grab | Cian=OR-Tal | Naranja=OR-Des | Rojo=IR',
        color='white', fontsize=11, fontweight='bold')
    ax.tick_params(colors='white'); ax.tick_params(axis='x', rotation=15)
    for spine in ax.spines.values(): spine.set_edgecolor('#333355')
    ax.grid(axis='y', ls='--', alpha=0.3, color='white', zorder=0)
    ax.legend(fontsize=9, facecolor='#1e2d40',
              edgecolor='#00d4ff', labelcolor='white')
    for i,v in enumerate(resultados_test):
        ax.text(i, v+1.5, f'{v:.1f}%', ha='center',
                color='white', fontsize=9.5, fontweight='bold', zorder=5)
    plt.tight_layout()
    plt.savefig(guardar, dpi=150, bbox_inches='tight', facecolor='#0d1b2a')
    plt.show(); print(f"  Barras LOBO: {guardar}")


# =============================================================================
# LOBO CNN
# =============================================================================
def ejecutar_lobo(X_all, Y_cls, Y_cod, top_idx):
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_pin = torch.cuda.is_available()
    print(f"\nDispositivo: {device}")

    # Subset de features seleccionadas por SOM
    X_top = X_all[:, top_idx]

    rods_presentes = set(np.unique(Y_cod))
    folds   = construir_folds(rods_presentes)
    n_folds = len(folds)
    print(f"\nConfig CNN: {EPOCAS} epocas | batch={BATCH_SIZE} | "
          f"lr={LR} | features={TOP_FEAT}\n")

    res_train=[]; res_test=[]; f1_folds=[]
    todas_preds=[]; todas_et=[]; hist_loss=[]
    et_eval=[]; prec_f=[]; rec_f=[]; fpr_f=[]

    for fold_info in tqdm(folds, desc="LOBO"):
        fold_id   = fold_info['fold_id']
        rod_test  = fold_info['test']
        trods     = fold_info['train']
        cls_test  = fold_info['clase_test']

        mtr = np.isin(Y_cod, trods)
        Xtr = X_top[mtr]; Ytr = Y_cls[mtr]
        if len(Xtr)==0: continue

        clases_p = np.unique(Ytr)
        tqdm.write(f"\n  [Fold {fold_id}] Test={rod_test} "
                   f"({NOMBRES_CLASE[cls_test]}) | Train={len(Xtr)}")

        sc  = RobustScaler()
        Xsc = sc.fit_transform(Xtr).astype(np.float32)
        Xtt = torch.tensor(Xsc[:,np.newaxis,:]); Ytt=torch.tensor(Ytr,dtype=torch.long)

        pw_np = compute_class_weight('balanced', classes=clases_p, y=Ytr)
        pw    = np.ones(N_CLASES, dtype=np.float32)
        for c,p in zip(clases_p, pw_np): pw[c]=p
        pw_t  = torch.tensor(pw).to(device)

        ld = DataLoader(TensorDataset(Xtt,Ytt), batch_size=BATCH_SIZE,
                        shuffle=True, drop_last=True,
                        pin_memory=use_pin, num_workers=0)

        modelo = CNN1D_Multiclass(n_features=TOP_FEAT,
                                  num_classes=N_CLASES).to(device)
        crit   = nn.CrossEntropyLoss(weight=pw_t)
        opt    = optim.AdamW(modelo.parameters(), lr=LR, weight_decay=1e-4)
        sch    = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCAS, eta_min=1e-5)

        modelo.train(); loss_ep_list=[]
        for ep in tqdm(range(EPOCAS), desc=f"    Fold {fold_id}", leave=False):
            lep=0.
            for bx,by in ld:
                bx,by=bx.to(device),by.to(device)
                opt.zero_grad(); loss=crit(modelo(bx),by)
                loss.backward()
                nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
                opt.step(); lep+=loss.item()
            loss_ep_list.append(lep/len(ld)); sch.step()
        hist_loss.append(loss_ep_list)

        modelo.eval(); cor=tot=0
        with torch.no_grad():
            for bx,by in ld:
                p=torch.argmax(modelo(bx.to(device)),dim=1)
                cor+=(p==by.to(device)).sum().item(); tot+=by.size(0)
        res_train.append(100*cor/tot)

        mrod = (Y_cod==rod_test)
        Xrod = X_top[mrod]; Yrod = Y_cls[mrod]
        if len(Xrod)==0: continue
        Xrsc = sc.transform(Xrod).astype(np.float32)
        Xrt  = torch.tensor(Xrsc[:,np.newaxis,:])
        Yrt  = torch.tensor(Yrod, dtype=torch.long)
        rld  = DataLoader(TensorDataset(Xrt,Yrt), batch_size=BATCH_SIZE,
                          shuffle=False, pin_memory=use_pin, num_workers=0)
        preds=[]
        modelo.eval()
        with torch.no_grad():
            for bx,by in rld:
                p=torch.argmax(modelo(bx.to(device)),dim=1).cpu().numpy()
                preds.extend(p); todas_preds.extend(p)
                todas_et.extend(by.numpy())

        acc_r = 100*np.mean(np.array(preds)==Yrod)
        f1_r  = f1_score(Yrod, preds, average='macro', zero_division=0)
        res_test.append(acc_r); f1_folds.append(f1_r); et_eval.append(rod_test)
        prec_f.append(precision_score(Yrod,preds,average=None,
                       labels=list(range(N_CLASES)),zero_division=0))
        rec_f.append(recall_score(Yrod,preds,average=None,
                      labels=list(range(N_CLASES)),zero_division=0))
        fpr_f.append(calcular_fpr(Yrod,preds))
        tqdm.write(f"    [{rod_test}] Acc {acc_r:.1f}%  F1 {f1_r:.3f}")

        del modelo; torch.cuda.empty_cache()

    # ── Resumen ───────────────────────────────────────────────────────────────
    target_names = [NOMBRES_CLASE[i] for i in range(N_CLASES)]
    acc_g   = np.mean(np.array(todas_preds)==np.array(todas_et))
    f1_g    = f1_score(todas_et, todas_preds, average='macro', zero_division=0)
    prec_g  = precision_score(todas_et, todas_preds, average=None,
                              labels=list(range(N_CLASES)), zero_division=0)
    rec_g   = recall_score(todas_et, todas_preds, average=None,
                           labels=list(range(N_CLASES)), zero_division=0)
    f1_cls  = f1_score(todas_et, todas_preds, average=None,
                       labels=list(range(N_CLASES)), zero_division=0)
    fpr_g   = calcular_fpr(todas_et, todas_preds)

    print(f"\n{'='*65}")
    print(f"  v12 CNN Top-{TOP_FEAT} (SOM-Pureza) | {len(res_test)} evaluaciones")
    print(f"  Acc Train  : {np.mean(res_train):.2f}% +/- {np.std(res_train):.2f}%")
    print(f"  Acc LOBO   : {np.mean(res_test):.2f}%  +/- {np.std(res_test):.2f}%")
    print(f"  Macro-F1   : {f1_g:.4f}")
    print(f"{'='*65}\n")
    print(classification_report(todas_et, todas_preds,
                                 target_names=target_names, digits=4))

    mg = {'acc_global':acc_g,'f1_macro':f1_g,'prec':prec_g,
          'rec':rec_g,'f1_por_clase':f1_cls,'fpr':fpr_g}

    graficar_tabla(mg, EXPERIMENTO, len(res_test), 'v12_tabla.png')
    graficar_accuracy_barras(res_test, et_eval, 'v12_lobo_acc.png')

    # Confusion
    fig,ax=plt.subplots(figsize=(9,7))
    sns.heatmap(confusion_matrix(todas_et,todas_preds),
                annot=True,fmt='d',cmap='Blues',ax=ax,
                xticklabels=target_names,yticklabels=target_names)
    ax.set_title(f'v12 Confusion LOBO | Top-{TOP_FEAT} SOM-Pureza',
                 fontweight='bold')
    ax.set_ylabel('Real'); ax.set_xlabel('Prediccion')
    plt.xticks(rotation=25,ha='right'); plt.tight_layout()
    plt.savefig('v12_confusion.png',dpi=150); plt.show()

    # Convergencia
    if hist_loss:
        lp=np.mean(hist_loss,axis=0); ls=np.std(hist_loss,axis=0)
        ep=range(1,EPOCAS+1)
        fig,ax=plt.subplots(figsize=(10,4))
        ax.plot(ep,lp,'#2196F3',lw=2,label='Loss promedio')
        ax.fill_between(ep,lp-ls,lp+ls,alpha=0.2,color='#2196F3',
                        label='Desv. std')
        ax.set_title(f'v12 Convergencia | Top-{TOP_FEAT} SOM-Pureza',
                     fontweight='bold')
        ax.set_xlabel('Epoca'); ax.set_ylabel('Loss')
        ax.legend(); ax.grid(ls='--',alpha=0.5)
        plt.tight_layout()
        plt.savefig('v12_convergencia.png',dpi=150); plt.show()

    print("\nArchivos v12:")
    print("  v12_tabla.png  | v12_lobo_acc.png")
    print("  v12_confusion.png | v12_convergencia.png")
    print("  v12_pureza_features.png")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\n" + "="*65)
    print("  v12 — SOM Pureza por Feature + CNN LOBO")
    print("="*65)

    # Cargar
    print("\nCargando datos...")
    X_all = np.load(RUTA_X).astype(np.float32)
    Y_cod = np.load(RUTA_COD, allow_pickle=True)
    names = list(np.load(RUTA_NAMES, allow_pickle=True))

    # Excluir EDM
    mask_exc = np.isin(Y_cod, list(EXCLUIDOS))
    X_all = X_all[~mask_exc]; Y_cod = Y_cod[~mask_exc]

    # Etiquetas CNN (5 clases)
    Y_cls = np.array([ETIQUETA_ROD.get(str(r),-1) for r in Y_cod],
                     dtype=np.int64)
    mask_v = Y_cls != -1
    X_all = X_all[mask_v]; Y_cod = Y_cod[mask_v]; Y_cls = Y_cls[mask_v]

    # Limpiar
    X_all, names = limpiar(X_all, names)
    print(f"  Shape: {X_all.shape}")

    # Escalar antes del SOM
    sc_pre = RobustScaler()
    X_sc   = sc_pre.fit_transform(X_all).astype(np.float32)

    # ── FASE 1: Seleccion de features por SOM-Pureza ──────────────────────────
    # Para el SOM de seleccion se usan TODOS los rodamientos (incl. EDM no
    # esta porque ya se excluyó, pero se usan las 5 clases CNN completas)
    top_idx, purezas = seleccionar_features_som(X_sc, Y_cls, names, TOP_FEAT)

    # Grafica de pureza
    graficar_barras_pureza(purezas, names, top_idx, 'v12_pureza_features.png')

    # ── FASE 2: CNN LOBO con las 30 features seleccionadas ────────────────────
    print(f"\n{'='*65}")
    print(f"  FASE 2 — CNN LOBO con Top-{TOP_FEAT} features (SOM-Pureza)")
    print(f"{'='*65}")
    ejecutar_lobo(X_sc, Y_cls, Y_cod, top_idx)


if __name__ == '__main__':
    main()
