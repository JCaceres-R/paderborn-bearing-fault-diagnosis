import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                            f1_score, accuracy_score)
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# SOM SEMI-SUPERVISADO v3 — Dataset Paderborn completo
# Entrenamiento: NO supervisado (X solamente, seleccion por VARIANZA sin Y)
# Visualizacion: semi-supervisado — etiquetas solo para evaluar y colorear
# =============================================================================

N_CLASES = 6

NOMBRES_CLASE = {
    0: 'Sano',
    1: 'OR-Grabado',
    2: 'OR-Taladro',
    3: 'OR-Desgaste',
    4: 'IR',
    5: 'EDM',
}

COLORES_CLASE = {
    0: '#42A5F5',
    1: '#66BB6A',
    2: '#26C6DA',
    3: '#FFA726',
    4: '#EF5350',
    5: '#AB47BC',
}

ETIQUETA_ROD = {
    'K001': 0, 'K002': 0, 'K003': 0, 'K004': 0, 'K005': 0, 'K006': 0,
    'KA03': 1, 'KA05': 1, 'KA06': 1,
    'KA07': 2, 'KA08': 2, 'KA09': 2,
    'KA04': 3, 'KA15': 3, 'KA16': 3, 'KA22': 3, 'KA30': 3,
    'KI03': 4, 'KI04': 4, 'KI05': 4, 'KI07': 4, 'KI08': 4,
    'KI14': 4, 'KI16': 4, 'KI17': 4, 'KI18': 4, 'KI21': 4,
    'KA01': 5, 'KI01': 5,
}

# ── Rutas ─────────────────────────────────────────────────────────────────────
RUTA_X     = r"\X_feat_crudos.npy"
RUTA_COD   = r"\Y_codigos_crudos.npy"
RUTA_NAMES = r"\feature_names_crudos.npy"

# ── Parametros SOM ────────────────────────────────────────────────────────────
SOM_ROWS   = 50
SOM_COLS   = 50
TOP_K      = 80
N_ITER     = 500000
LR_INI     = 0.5
LR_FIN     = 0.01
SIGMA_INI  = max(SOM_ROWS, SOM_COLS) / 2.0
SIGMA_FIN  = 1.0
SUBSAMPLE  = 800000     # muestras para entrenar
SEED       = 42


# =============================================================================
# CLASE SOM VECTORIZADA
# =============================================================================
class SOM:
    def __init__(self, rows, cols, n_features, seed=42):
        self.rows       = rows
        self.cols       = cols
        self.n_features = n_features
        self.n_neu      = rows * cols
        rng = np.random.default_rng(seed)

        # Posiciones 2D de cada neurona: (n_neu, 2)
        rc = np.array([[r, c] for r in range(rows) for c in range(cols)],
                      dtype=np.float32)
        self.pos = rc

        # Pesos: (n_neu, n_features)
        self.W = rng.standard_normal((self.n_neu, n_features)).astype(np.float32)

        # Distancias al cuadrado entre todas las neuronas (precalculadas)
        d = self.pos[:, None, :] - self.pos[None, :, :]   # (N,N,2)
        self._d2 = (d ** 2).sum(axis=2)                   # (N,N)

    def _vecindad_batch(self, bmu_idx, sigma):
        """Vecindad gaussiana para un BMU: shape (n_neu,)"""
        d2  = self._d2[bmu_idx]                           # (n_neu,)
        return np.exp(-d2 / (2 * sigma ** 2)).astype(np.float32)

    def entrenar(self, X, n_iter, lr_ini, lr_fin, sigma_ini, sigma_fin):
        N = len(X)
        rng = np.random.default_rng(SEED)

        # Inicializar pesos en rango de datos
        x_min = X.min(axis=0); x_max = X.max(axis=0)
        self.W = (rng.random((self.n_neu, self.n_features)).astype(np.float32)
                  * (x_max - x_min) + x_min)

        tau_sigma = n_iter / np.log(sigma_ini / sigma_fin + 1e-9)

        for it in tqdm(range(n_iter), desc="Entrenando SOM", miniters=n_iter//200):
            x = X[rng.integers(0, N)]                      # muestra aleatoria

            # Decaimiento
            alpha = lr_ini  * (lr_fin  / lr_ini) ** (it / n_iter)
            sigma = sigma_ini * (sigma_fin / sigma_ini) ** (it / n_iter)

            # Competicion: BMU — distancia euclidea al cuadrado
            diff = self.W - x                              # (n_neu, D)
            dist2 = (diff ** 2).sum(axis=1)                # (n_neu,)
            bmu   = int(np.argmin(dist2))

            # Cooperacion: vecindad gaussiana
            h = self._vecindad_batch(bmu, sigma)[:, None]  # (n_neu, 1)

            # Adaptacion: w = w + alpha * h * (x - w)
            self.W += alpha * h * (x - self.W)

    def mapear(self, X):
        """BMU para cada muestra — vectorizado por lotes."""
        bmus = np.empty(len(X), dtype=int)
        BS   = 2048
        for i in range(0, len(X), BS):
            Xb   = X[i:i+BS]                              # (B, D)
            diff = Xb[:, None, :] - self.W[None, :, :]   # (B, N, D)
            d2   = (diff**2).sum(axis=2)                  # (B, N)
            bmus[i:i+BS] = d2.argmin(axis=1)
        return bmus   # indices planos: bmu_idx = r*cols + c

    def idx_a_rc(self, idx):
        return idx // self.cols, idx % self.cols


# =============================================================================
# METRICAS DE CLASIFICACION
# =============================================================================
def calcular_metricas(bmus_idx, Y_cls, som):
    """
    Para cada muestra, su prediccion es la clase mayoritaria de su BMU.
    Devuelve predicciones y diccionario de metricas.
    """
    rows, cols = som.rows, som.cols
    n_neu = rows * cols

    # Votos por neurona
    votos = np.zeros((n_neu, N_CLASES), dtype=int)
    for idx, cls in zip(bmus_idx, Y_cls):
        votos[idx, cls] += 1

    # Clase mayoritaria por neurona
    clase_neu = np.argmax(votos, axis=1)          # (n_neu,)
    hits_neu  = votos.sum(axis=1)                 # (n_neu,)

    # Pureza por neurona
    pureza_neu = np.zeros(n_neu)
    activas = hits_neu > 0
    pureza_neu[activas] = (votos[activas, clase_neu[activas]] /
                           hits_neu[activas].astype(float))

    # Prediccion por muestra = clase mayoritaria de su BMU
    y_pred = clase_neu[bmus_idx]

    acc     = accuracy_score(Y_cls, y_pred)
    f1_mac  = f1_score(Y_cls, y_pred, average='macro', zero_division=0)
    f1_cls  = f1_score(Y_cls, y_pred, average=None,
                       labels=list(range(N_CLASES)), zero_division=0)

    metricas = {
        'acc':          acc,
        'f1_macro':     f1_mac,
        'f1_por_clase': f1_cls,
        'votos':        votos,
        'clase_neu':    clase_neu,
        'hits_neu':     hits_neu,
        'pureza_neu':   pureza_neu,
        'y_pred':       y_pred,
        'y_true':       Y_cls,
    }
    return metricas


# =============================================================================
# VISUALIZACIONES
# =============================================================================
def plot_hit_map(metricas, som, guardar='som_hitmap.png'):
    rows, cols = som.rows, som.cols
    clase_neu  = metricas['clase_neu'].reshape(rows, cols)
    hits_neu   = metricas['hits_neu'].reshape(rows, cols)
    pureza_neu = metricas['pureza_neu'].reshape(rows, cols)
    max_hits   = hits_neu.max() if hits_neu.max() > 0 else 1

    fig, ax = plt.subplots(figsize=(13, 12))
    fig.patch.set_facecolor('#0d1b2a')
    ax.set_facecolor('#0d1b2a')

    for r in range(rows):
        for c in range(cols):
            n_hits = hits_neu[r, c]
            cls    = clase_neu[r, c]
            color  = COLORES_CLASE[cls] if n_hits > 0 else '#1a1a2e'
            size   = (0.3 + 0.65 * (n_hits / max_hits)) if n_hits > 0 else 0.2
            alpha  = (0.3 + 0.7 * pureza_neu[r, c]) if n_hits > 0 else 0.15
            circ   = plt.Circle((c, rows-1-r), size/2,
                                color=color, alpha=alpha, zorder=2)
            ax.add_patch(circ)

    ax.set_xlim(-0.8, cols-0.2); ax.set_ylim(-0.8, rows-0.2)
    ax.set_aspect('equal')
    ax.set_xticks(range(cols)); ax.set_yticks(range(rows))
    ax.tick_params(colors='#aaaaaa', labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')

    parches = [mpatches.Patch(color=COLORES_CLASE[c],
                              label=NOMBRES_CLASE[c]) for c in range(N_CLASES)]
    ax.legend(handles=parches, loc='upper right', fontsize=9,
              facecolor='#1e2d40', edgecolor='#00d4ff', labelcolor='white')

    acc    = metricas['acc']
    f1_mac = metricas['f1_macro']
    ax.set_title(
        f'SOM Semi-Supervisado — Hit Map\n'
        f'Grilla {rows}x{cols} | Top-{TOP_K} features | '
        f'{N_ITER:,} iteraciones\n'
        f'Accuracy: {acc:.3f}  |  Macro-F1: {f1_mac:.3f}',
        color='white', fontsize=11, fontweight='bold', pad=10)
    ax.set_xlabel('Columna', color='#aaaaaa')
    ax.set_ylabel('Fila',    color='#aaaaaa')

    plt.tight_layout()
    plt.savefig(guardar, dpi=150, bbox_inches='tight', facecolor='#0d1b2a')
    plt.show()
    print(f"  Hit map guardado: {guardar}")


def plot_confusion(Y_cls, y_pred, guardar='som_confusion.png'):
    target_names = [NOMBRES_CLASE[i] for i in range(N_CLASES)]
    cm = confusion_matrix(Y_cls, y_pred)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=target_names, yticklabels=target_names)
    ax.set_title('Matriz de Confusion — SOM Semi-Supervisado',
                 fontweight='bold', fontsize=12)
    ax.set_ylabel('Etiqueta Real')
    ax.set_xlabel('Prediccion SOM')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(guardar, dpi=150)
    plt.show()
    print(f"  Confusion guardada: {guardar}")


def plot_metricas_tabla(metricas, guardar='som_tabla.png'):
    """Tabla de metricas por clase al estilo del proyecto."""
    BG_HEADER = '#1a3a5c'; BG_EXP   = '#1e2d40'
    BG_ROW_1  = '#1e2d40'; BG_ROW_2 = '#243448'
    FG_HEADER = '#00d4ff'; FG_WHITE = '#ffffff'; FG_LIGHT = '#cce8ff'

    cols_t     = ['Metodo', 'Clase', 'Accuracy', 'F1-Score']
    col_widths = [0.22, 0.22, 0.22, 0.22]

    acc    = metricas['acc']
    f1_mac = metricas['f1_macro']
    f1_cls = metricas['f1_por_clase']

    n_rows = 1 + N_CLASES
    row_h  = 0.52
    fig_h  = row_h * n_rows + 0.6

    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.set_xlim(0, 0.88); ax.set_ylim(0, 1); ax.axis('off')
    fig.patch.set_facecolor('#0d1b2a')

    x_starts   = np.cumsum([0] + col_widths[:-1])
    row_height = 1.0 / (n_rows + 0.5)

    def draw_cell(ax, x, y, w, h, text, bg, fg, fontsize=9, bold=False):
        rect = mpatches.FancyBboxPatch(
            (x+0.003, y-h+0.008), w-0.006, h-0.01,
            boxstyle="round,pad=0.005", linewidth=0, facecolor=bg, zorder=1)
        ax.add_patch(rect)
        ax.text(x+w/2, y-h/2, text, ha='center', va='center',
                fontsize=fontsize, color=fg,
                fontweight='bold' if bold else 'normal', zorder=2)

    y_hdr = 1.0
    for j, (col, w) in enumerate(zip(cols_t, col_widths)):
        draw_cell(ax, x_starts[j], y_hdr, w, row_height,
                  col, BG_HEADER, FG_HEADER, fontsize=9.5, bold=True)

    for ci in range(N_CLASES):
        y_row = y_hdr - (ci+1)*row_height
        bg    = BG_ROW_1 if ci % 2 == 0 else BG_ROW_2

        if ci == 0:
            draw_cell(ax, x_starts[0], y_hdr-row_height, col_widths[0],
                      row_height*N_CLASES,
                      f"SOM {SOM_ROWS}x{SOM_COLS}\n"
                      f"Top-{TOP_K} | {N_ITER:,} iter\n"
                      f"Acc={acc:.3f} | F1={f1_mac:.3f}",
                      BG_EXP, FG_LIGHT, fontsize=8)
            draw_cell(ax, x_starts[2], y_hdr-row_height, col_widths[2],
                      row_height*N_CLASES,
                      f"{acc:.3f}", BG_EXP, FG_WHITE, fontsize=9)

        draw_cell(ax, x_starts[1], y_row, col_widths[1], row_height,
                  NOMBRES_CLASE[ci], bg, FG_WHITE, fontsize=8.5)
        draw_cell(ax, x_starts[3], y_row, col_widths[3], row_height,
                  f"{f1_cls[ci]:.3f}", bg, FG_WHITE, fontsize=9)

    plt.tight_layout(pad=0.1)
    plt.savefig(guardar, dpi=180, bbox_inches='tight', facecolor='#0d1b2a')
    plt.show()
    print(f"  Tabla guardada: {guardar}")


def plot_accuracy_barras(metricas, guardar='som_accuracy_barras.png'):
    """
    Diagrama de barras: accuracy por clase + accuracy global.
    Accuracy por clase = recall (fraccion correctamente clasificada).
    """
    from sklearn.metrics import recall_score
    Y_cls_arr = metricas['y_true']
    y_pred    = metricas['y_pred']

    rec_cls = recall_score(Y_cls_arr, y_pred, average=None,
                           labels=list(range(N_CLASES)), zero_division=0)
    acc_global = metricas['acc']

    nombres = [NOMBRES_CLASE[i] for i in range(N_CLASES)]
    colores = [COLORES_CLASE[i] for i in range(N_CLASES)]

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor('#0d1b2a')
    ax.set_facecolor('#0d1b2a')

    bars = ax.bar(nombres, rec_cls * 100, color=colores,
                  edgecolor='white', linewidth=0.8, zorder=3)

    # Linea de accuracy global
    ax.axhline(acc_global * 100, color='white', ls='--', lw=1.8,
               label=f'Accuracy global: {acc_global*100:.2f}%', zorder=4)

    # Linea de azar
    ax.axhline(100 / N_CLASES, color='#FF6B6B', ls=':', lw=1.5,
               label=f'Azar ({100/N_CLASES:.1f}%)', zorder=4)

    ax.set_ylim(0, 115)
    ax.set_ylabel('Accuracy por clase (%)', color='white', fontsize=11)
    ax.set_xlabel('Clase', color='white', fontsize=11)
    ax.set_title(
        f'SOM Semi-Supervisado — Accuracy por clase\n'
        f'Grilla {SOM_ROWS}x{SOM_COLS} | Top-{TOP_K} varianza | '
        f'{N_ITER:,} iteraciones | Acc global: {acc_global*100:.2f}%',
        color='white', fontsize=11, fontweight='bold')

    ax.tick_params(colors='white', labelsize=10)
    ax.tick_params(axis='x', rotation=15)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')
    ax.grid(axis='y', ls='--', alpha=0.3, color='white', zorder=0)
    ax.legend(fontsize=9, facecolor='#1e2d40',
              edgecolor='#00d4ff', labelcolor='white')

    for bar, val in zip(bars, rec_cls):
        ax.text(bar.get_x() + bar.get_width()/2,
                val*100 + 1.5, f'{val*100:.1f}%',
                ha='center', va='bottom', color='white',
                fontsize=10, fontweight='bold', zorder=5)

    plt.tight_layout()
    plt.savefig(guardar, dpi=150, bbox_inches='tight', facecolor='#0d1b2a')
    plt.show()
    print(f"  Barras accuracy guardado: {guardar}")

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(U, cmap='bone_r', interpolation='bicubic',
                   origin='upper', aspect='equal')
    plt.colorbar(im, ax=ax, label='Distancia promedio a vecinos')
    ax.set_title(
        f'U-Matrix — SOM {som.rows}x{som.cols}\n'
        f'Accuracy: {metricas["acc"]:.3f}  |  Macro-F1: {metricas["f1_macro"]:.3f}\n'
        'Oscuro = frontera entre grupos | Claro = region homogenea',
        fontweight='bold')
    ax.set_xlabel('Columna'); ax.set_ylabel('Fila')
    plt.tight_layout()
    plt.savefig(guardar, dpi=150)
    plt.show()
    print(f"  U-Matrix guardada: {guardar}")


def plot_clase_maps(bmus_idx, Y_cls, som, guardar='som_clases.png'):
    from matplotlib.colors import LinearSegmentedColormap
    rows, cols = som.rows, som.cols
    densidad   = np.zeros((N_CLASES, rows, cols), dtype=float)
    for idx, cls in zip(bmus_idx, Y_cls):
        r, c = idx // cols, idx % cols
        densidad[cls, r, c] += 1

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.patch.set_facecolor('#0d1b2a')

    f1_cls = [f1_score(Y_cls == ci, bmus_idx == -999,   # calculado aparte
                       average='binary', zero_division=0)
              for ci in range(N_CLASES)]

    for ci, ax in enumerate(axes.flat):
        if ci >= N_CLASES:
            ax.set_visible(False); continue
        d = densidad[ci]
        if d.max() > 0: d = d / d.max()
        cmap = LinearSegmentedColormap.from_list(
            '', ['#0d1b2a', COLORES_CLASE[ci]])
        im = ax.imshow(d, cmap=cmap, interpolation='bicubic',
                       origin='upper', aspect='equal', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(NOMBRES_CLASE[ci], color='white',
                     fontsize=11, fontweight='bold')
        ax.set_facecolor('#0d1b2a')
        ax.tick_params(colors='#aaaaaa', labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor('#333355')

    fig.suptitle(
        f'Densidad por clase — SOM {rows}x{cols} | Top-{TOP_K} features',
        color='white', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(guardar, dpi=150, bbox_inches='tight', facecolor='#0d1b2a')
    plt.show()
    print(f"  Mapa por clases guardado: {guardar}")


def plot_pureza(metricas, som, guardar='som_pureza.png'):
    rows, cols = som.rows, som.cols
    hits   = metricas['hits_neu'].reshape(rows, cols).astype(float)
    pureza = metricas['pureza_neu'].reshape(rows, cols)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    im1 = axes[0].imshow(np.log1p(hits), cmap='plasma',
                          interpolation='bicubic', origin='upper', aspect='equal')
    plt.colorbar(im1, ax=axes[0], label='log(1 + hits)')
    axes[0].set_title('Densidad de activaciones (log)', fontweight='bold')
    axes[0].set_xlabel('Columna'); axes[0].set_ylabel('Fila')

    pureza_m = np.where(hits > 0, pureza, np.nan)
    im2 = axes[1].imshow(pureza_m, cmap='RdYlGn', interpolation='bicubic',
                          origin='upper', aspect='equal', vmin=0, vmax=1)
    plt.colorbar(im2, ax=axes[1], label='Pureza')
    axes[1].set_title('Pureza por neurona', fontweight='bold')
    axes[1].set_xlabel('Columna'); axes[1].set_ylabel('Fila')

    activas = hits[hits > 0]
    plt.suptitle(
        f'Analisis de calidad — SOM {rows}x{cols}\n'
        f'Pureza media: {metricas["pureza_neu"][metricas["hits_neu"]>0].mean():.3f}  |  '
        f'Neuronas puras (>0.9): '
        f'{(metricas["pureza_neu"][metricas["hits_neu"]>0]>0.9).sum()}',
        fontweight='bold', fontsize=11)
    plt.tight_layout()
    plt.savefig(guardar, dpi=150)
    plt.show()
    print(f"  Pureza guardada: {guardar}")


# =============================================================================
# MAIN
# =============================================================================
def ejecutar_som():
    # ── Cargar ────────────────────────────────────────────────────────────────
    print("\nCargando datos...")
    X_all  = np.load(RUTA_X).astype(np.float32)
    Y_cod  = np.load(RUTA_COD, allow_pickle=True)
    names  = list(np.load(RUTA_NAMES, allow_pickle=True))
    print(f"  Shape original: {X_all.shape}")

    # ── Etiquetas ─────────────────────────────────────────────────────────────
    Y_cls = np.array([ETIQUETA_ROD.get(str(r), -1) for r in Y_cod], dtype=np.int64)
    mask  = Y_cls != -1
    X_all = X_all[mask]; Y_cls = Y_cls[mask]

    # ── Limpieza ──────────────────────────────────────────────────────────────
    X_all = np.nan_to_num(X_all, nan=0., posinf=0., neginf=0.)
    mv    = np.var(X_all, axis=0) > 1e-15
    X_all = X_all[:, mv]
    names = [n for n, m in zip(names, mv) if m]

    print(f"\n  Distribucion de clases:")
    for c in range(N_CLASES):
        print(f"    {c} ({NOMBRES_CLASE[c]:14s}): {(Y_cls==c).sum():6d} ventanas")

    # ── Seleccion features — PURAMENTE NO SUPERVISADA (varianza) ─────────────
    # No se usan las etiquetas Y en ningun paso del entrenamiento.
    # Se seleccionan las Top-K features con mayor varianza en X.
    print(f"\n  Seleccionando Top-{TOP_K} features por VARIANZA (no supervisado)...")
    varianzas = np.var(X_all, axis=0)
    top_idx   = np.argsort(varianzas)[::-1][:TOP_K]
    X_top     = X_all[:, top_idx]
    print(f"  Top-5: {[names[i] for i in top_idx[:5]]}")

    # ── Escalado ──────────────────────────────────────────────────────────────
    scaler = RobustScaler()
    X_sc   = scaler.fit_transform(X_top).astype(np.float32)

    # ── Submuestreo para entrenamiento ────────────────────────────────────────
    rng = np.random.default_rng(SEED)
    N   = len(X_sc)
    if N > SUBSAMPLE:
        idx_s   = rng.choice(N, SUBSAMPLE, replace=False)
        X_train = X_sc[idx_s]
        print(f"\n  Submuestreo entrenamiento: {SUBSAMPLE:,} / {N:,}")
    else:
        X_train = X_sc

    # ── Entrenar SOM ──────────────────────────────────────────────────────────
    print(f"\n  SOM {SOM_ROWS}x{SOM_COLS} | {N_ITER:,} iter | "
          f"lr: {LR_INI}->{LR_FIN} | sigma: {SIGMA_INI:.1f}->{SIGMA_FIN:.1f}")
    som = SOM(SOM_ROWS, SOM_COLS, TOP_K, seed=SEED)
    som.entrenar(X_train, N_ITER, LR_INI, LR_FIN, SIGMA_INI, SIGMA_FIN)

    # ── Mapear TODAS las muestras ─────────────────────────────────────────────
    print(f"\n  Mapeando {N:,} muestras...")
    bmus_idx = som.mapear(X_sc)     # indices planos

    # ── Metricas ──────────────────────────────────────────────────────────────
    metricas = calcular_metricas(bmus_idx, Y_cls, som)

    print(f"\n{'='*65}")
    print(f"  METRICAS DE CLASIFICACION — SOM {SOM_ROWS}x{SOM_COLS}")
    print(f"{'='*65}")
    print(f"  Accuracy global : {metricas['acc']:.4f}  "
          f"({metricas['acc']*100:.2f}%)")
    print(f"  Macro-F1        : {metricas['f1_macro']:.4f}")
    print(f"\n  F1 por clase:")
    for c in range(N_CLASES):
        print(f"    {c} ({NOMBRES_CLASE[c]:14s}): {metricas['f1_por_clase'][c]:.4f}")

    n_act   = (metricas['hits_neu'] > 0).sum()
    pur_act = metricas['pureza_neu'][metricas['hits_neu'] > 0]
    print(f"\n  Neuronas activas     : {n_act} / {SOM_ROWS*SOM_COLS}")
    print(f"  Pureza media         : {pur_act.mean():.3f}")
    print(f"  Neuronas puras (>0.9): {(pur_act > 0.9).sum()} "
          f"({100*(pur_act>0.9).mean():.1f}%)")
    print(f"  Neuronas mixtas(<0.6): {(pur_act < 0.6).sum()} "
          f"({100*(pur_act<0.6).mean():.1f}%)")
    print(f"{'='*65}\n")

    target_names = [NOMBRES_CLASE[i] for i in range(N_CLASES)]
    print(classification_report(Y_cls, metricas['y_pred'],
                                 target_names=target_names, digits=4))

    # ── Figuras ───────────────────────────────────────────────────────────────
    plot_hit_map(metricas, som,                      'som_hitmap.png')
    plot_confusion(Y_cls, metricas['y_pred'],        'som_confusion.png')
    plot_metricas_tabla(metricas,                    'som_tabla.png')
    plot_accuracy_barras(metricas,                   'som_accuracy_barras.png')
    plot_umatrix(som, metricas,                      'som_umatrix.png')
    plot_clase_maps(bmus_idx, Y_cls, som,            'som_clases.png')
    plot_pureza(metricas, som,                       'som_pureza.png')

    print("\nArchivos guardados:")
    print("  som_hitmap.png         | som_confusion.png")
    print("  som_tabla.png          | som_accuracy_barras.png")
    print("  som_umatrix.png        | som_clases.png")
    print("  som_pureza.png")


if __name__ == '__main__':
    ejecutar_som()
