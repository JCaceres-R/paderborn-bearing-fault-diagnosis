import sys
sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm


# =============================================================================
# ARQUITECTURA CNN 1D CON FFT LOGARÍTMICA — CORREGIDA
# =============================================================================
# ¿Por qué FFT? En vez de darle la señal cruda a la CNN, primero la convertimos
# al dominio de frecuencias. Las fallas en rodamientos generan frecuencias
# características específicas (BPFI, BPFO, BSF), por lo que el espectro de
# frecuencias es una representación más directa del problema que la señal cruda.
#
# ¿Por qué logarítmica? El espectro tiene componentes con amplitudes muy distintas
# (algunas enormes, otras minúsculas). El logaritmo comprime ese rango dinámico
# y hace que la CNN pueda ver tanto las frecuencias dominantes como las débiles.
# =============================================================================
class MCSA_CNN1D_FFT(nn.Module):
    def __init__(self, in_channels=2, num_classes=3, window_size=4096):
        super(MCSA_CNN1D_FFT, self).__init__()

        # La FFT de una señal real de N puntos produce N//2 + 1 componentes
        # Para window_size=4096 → 2049 puntos en el espectro
        fft_size = window_size // 2 + 1  # = 2049

        # --- Bloque 1 ---
        # kernel_size=16 es apropiado para el espectro (señal más compacta que raw)
        # padding=8 = kernel_size//2 → padding simétrico correcto
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=16, stride=1, padding=8)
        self.bn1   = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)

        # --- Bloque 2 ---
        # padding=4 = kernel_size//2 para kernel_size=8
        self.conv2 = nn.Conv1d(32, 64, kernel_size=8, stride=1, padding=4)
        self.bn2   = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)

        # --- Bloque 3 (NUEVO vs versión original) ---
        # Capa extra para aprender representaciones más abstractas
        self.conv3 = nn.Conv1d(64, 128, kernel_size=4, stride=1, padding=2)
        self.bn3   = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=4, stride=4)

        # Calcular tamaño del flatten pasando por el flujo COMPLETO (FFT incluida)
        with torch.no_grad():
            dummy    = torch.zeros(1, in_channels, window_size)
            out      = self._forward_convs(dummy)
            self.flatten_size = out.view(1, -1).size(1)

        # --- Capas Densas ---
        self.fc1      = nn.Linear(self.flatten_size, 256)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2      = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(p=0.3)
        self.fc3      = nn.Linear(128, num_classes)

    def _aplicar_fft(self, x):
        """
        Convierte la señal cruda al espectro de magnitud logarítmico.
        Entrada:  (batch, canales, 4096) — señal en tiempo
        Salida:   (batch, canales, 2049) — magnitud log del espectro
        """
        espectro = torch.fft.rfft(x, dim=2)          # → complejo (batch, C, 2049)
        magnitud = torch.abs(espectro)                # → real     (batch, C, 2049)
        log_mag  = torch.log(magnitud + 1e-8)         # → log      (batch, C, 2049)
        return log_mag

    def _forward_convs(self, x):
        """Flujo convolucional completo incluyendo FFT (para calcular flatten_size)."""
        x = self._aplicar_fft(x)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        return x

    def forward(self, x):
        x = self._forward_convs(x)
        x = x.view(x.size(0), -1)
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


# =============================================================================
# RODAMIENTOS DEL PAPER (Tabla 10 de Lessmeier et al.)
# =============================================================================
RODAMIENTOS_TABLA_10 = [
    'K001', 'K002', 'K003', 'K004', 'K005',  # Sanos        → clase 0
    'KA04', 'KA15', 'KA16', 'KA22', 'KA30',  # Daño Externo → clase 2
    'KI04', 'KI14', 'KI16', 'KI18', 'KI21',  # Daño Interno → clase 1
]


# =============================================================================
# CARGA Y FILTRADO DE DATOS — CON MEMORY MAP
# =============================================================================
def cargar_subconjunto_paper(ruta_x, ruta_y):
    print("Cargando y filtrando dataset (Tabla 10 de Lessmeier)...")

    # mmap_mode='r' → lee del disco bajo demanda, no carga los 4GB en RAM de golpe
    X_completo = np.load(ruta_x, mmap_mode='r')
    Y_nombres  = np.load(ruta_y)

    # Primero recolectamos solo los índices válidos (sin tocar X aún)
    indices, Y_clases, Y_codigos = [], [], []

    for i, nombre in enumerate(Y_nombres):
        codigo = nombre.split('_')[3]
        if codigo in RODAMIENTOS_TABLA_10:
            if   codigo.startswith('K00'): clase = 0
            elif codigo.startswith('KI'):  clase = 1
            elif codigo.startswith('KA'):  clase = 2

            indices.append(i)
            Y_clases.append(clase)
            Y_codigos.append(codigo)

    # Extraer solo las filas necesarias en float32 (mitad de memoria que float64)
    indices = np.array(indices)
    X = np.array(X_completo[indices], dtype=np.float32)
    Y = np.array(Y_clases)
    C = np.array(Y_codigos)

    del X_completo, Y_nombres  # Liberar el mmap

    # Reporte de distribución
    clases, conteos = np.unique(Y, return_counts=True)
    nombres_clase = {0: 'Sano', 1: 'Daño Interno', 2: 'Daño Externo'}
    print("\nDistribución de clases:")
    for c, n in zip(clases, conteos):
        print(f"  Clase {c} ({nombres_clase[c]}): {n} ventanas ({100*n/len(Y):.1f}%)")

    return X, Y, C


# =============================================================================
# NORMALIZACIÓN POR CANAL — CORRECCIÓN PRINCIPAL
# =============================================================================
def normalizar_por_canal(X_train, X_test):
    """
    Normaliza cada canal de forma independiente.

    X tiene shape (N, 2, 4096):
      axis (0,2) = sobre muestras y tiempo → resultado shape (1, 2, 1)
    Así cada canal tiene su propia media y desviación estándar.
    El test siempre se normaliza con las estadísticas del train (sin trampa).
    """
    media = X_train.mean(axis=(0, 2), keepdims=True)  # (1, 2, 1)
    std   = X_train.std(axis=(0, 2),  keepdims=True)  # (1, 2, 1)

    X_train_norm = (X_train - media) / (std + 1e-8)
    X_test_norm  = (X_test  - media) / (std + 1e-8)

    return X_train_norm, X_test_norm


# =============================================================================
# BUCLE LOBO — CON TODAS LAS CORRECCIONES
# =============================================================================
def ejecutar_lobo_oficial():
    ruta_x = r"C:\Users\USER\Desktop\IC2\proyecto\Dataset_Ventaneado\X_ventanas_4096.npy"
    ruta_y = r"C:\Users\USER\Desktop\IC2\proyecto\Dataset_Ventaneado\Y_ventanas_4096.npy"

    # --- Detectar GPU ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        vram   = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\nGPU DISPONIBLE: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM Total: {vram:.2f} GB")
        use_pin_memory = True
    else:
        device = torch.device("cpu")
        print("\nGPU NO DISPONIBLE - Usando CPU")
        use_pin_memory = False

    X_all, Y_cls, Y_cod = cargar_subconjunto_paper(ruta_x, ruta_y)

    # --- Hiperparámetros ---
    EPOCAS     = 60    # Más que el original (30) pero menos que nuestra versión raw (80)
                       # El espectro FFT es más informativo → converge más rápido
    BATCH_SIZE = 128
    LR_INICIAL = 0.001

    resultados_train = []
    resultados_test  = []

    # Para las evidencias visuales
    todas_predicciones    = []
    todas_etiquetas_reales = []
    historial_loss        = []

    print(f"\nIniciando LOBO Cross-Validation ({len(RODAMIENTOS_TABLA_10)} rodamientos)...")
    print(f"Modelo: CNN 1D + FFT Logarítmica (corregida)")
    print(f"Config: {EPOCAS} épocas | batch={BATCH_SIZE} | lr={LR_INICIAL}\n")

    for idx, rod_test in enumerate(tqdm(RODAMIENTOS_TABLA_10, desc="LOBO Progress")):

        # ----- Separar Train / Test -----
        mask_test  = (Y_cod == rod_test)
        mask_train = ~mask_test

        X_train_np = X_all[mask_train]
        Y_train_np = Y_cls[mask_train]
        X_test_np  = X_all[mask_test]
        Y_test_np  = Y_cls[mask_test]

        # ----- CORRECCIÓN 1: Normalización por canal -----
        X_train_np, X_test_np = normalizar_por_canal(X_train_np, X_test_np)

        X_train_t = torch.tensor(X_train_np, dtype=torch.float32)
        Y_train_t = torch.tensor(Y_train_np, dtype=torch.long)
        X_test_t  = torch.tensor(X_test_np,  dtype=torch.float32)
        Y_test_t  = torch.tensor(Y_test_np,  dtype=torch.long)

        # ----- CORRECCIÓN 2: Pesos de clase contra el desbalance -----
        clases_unicas = np.unique(Y_train_np)
        pesos_np = compute_class_weight(
            class_weight='balanced',
            classes=clases_unicas,
            y=Y_train_np
        )
        pesos_completos = np.ones(3)
        for c, p in zip(clases_unicas, pesos_np):
            pesos_completos[c] = p
        pesos_tensor = torch.tensor(pesos_completos, dtype=torch.float32).to(device)

        # DataLoaders
        train_loader = DataLoader(
            TensorDataset(X_train_t, Y_train_t),
            batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
            pin_memory=use_pin_memory, num_workers=0
        )
        test_loader = DataLoader(
            TensorDataset(X_test_t, Y_test_t),
            batch_size=BATCH_SIZE, shuffle=False,
            pin_memory=use_pin_memory, num_workers=0
        )

        # ----- Modelo, loss y optimizador -----
        modelo      = MCSA_CNN1D_FFT().to(device)
        criterio    = nn.CrossEntropyLoss(weight=pesos_tensor)  # Con pesos de clase
        optimizador = optim.Adam(modelo.parameters(), lr=LR_INICIAL, weight_decay=1e-4)

        # CORRECCIÓN 3: Scheduler — LR se divide a la mitad cada 20 épocas
        scheduler = optim.lr_scheduler.StepLR(optimizador, step_size=20, gamma=0.5)

        # ----- Entrenamiento -----
        modelo.train()
        loss_rodamiento = []
        for epoca in range(EPOCAS):
            loss_epoca = 0.0
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizador.zero_grad()
                perdida = criterio(modelo(bx), by)
                perdida.backward()
                optimizador.step()
                loss_epoca += perdida.item()
            loss_rodamiento.append(loss_epoca / len(train_loader))
            scheduler.step()
        historial_loss.append(loss_rodamiento)

        # ----- Evaluación Train -----
        modelo.eval()
        correctos_tr, total_tr = 0, 0
        with torch.no_grad():
            for bx, by in train_loader:
                preds = torch.argmax(modelo(bx.to(device)), dim=1)
                correctos_tr += (preds == by.to(device)).sum().item()
                total_tr     += by.size(0)
        acc_train = 100 * correctos_tr / total_tr
        resultados_train.append(acc_train)

        # ----- Evaluación Test (LOBO) -----
        correctos_te, total_te = 0, 0
        with torch.no_grad():
            for bx, by in test_loader:
                preds = torch.argmax(modelo(bx.to(device)), dim=1)
                correctos_te += (preds == by.to(device)).sum().item()
                total_te     += by.size(0)
                # Acumular para métricas globales
                todas_predicciones.extend(preds.cpu().numpy())
                todas_etiquetas_reales.extend(by.numpy())
        acc_test = 100 * correctos_te / total_te
        resultados_test.append(acc_test)

        tqdm.write(f"  [{idx+1:02d}/15] Rodamiento test={rod_test:5s} | "
                   f"Train: {acc_train:5.1f}% | LOBO Test: {acc_test:5.1f}%")

        # Liberar memoria al final de cada iteración
        del modelo, X_train_t, X_test_t, Y_train_t, Y_test_t
        del train_loader, test_loader
        torch.cuda.empty_cache()

    # ----- Resultados finales -----
    acc_train_final = np.mean(resultados_train)
    acc_test_final  = np.mean(resultados_test)

    print(f"\n{'='*55}")
    print(f"  RESULTADO FINAL — CNN 1D + FFT Log (Corregida)")
    print(f"  Acc. Train promedio : {acc_train_final:.2f}%")
    print(f"  Acc. Test  promedio : {acc_test_final:.2f}%")
    print(f"{'='*55}\n")

    # =========================================================
    # EVIDENCIAS VISUALES
    # =========================================================
    nombres_clases = ['Sano', 'Dano Interno', 'Dano Externo']

    # --- Evidencia 1: Reporte de clasificacion global ---
    print("\n" + "="*55)
    print("  REPORTE DE CLASIFICACION GLOBAL (LOBO)")
    print("="*55)
    reporte = classification_report(
        todas_etiquetas_reales, todas_predicciones,
        target_names=nombres_clases, digits=4
    )
    print(reporte)

    # --- Evidencia 2: Matriz de confusion global ---
    matriz = confusion_matrix(todas_etiquetas_reales, todas_predicciones)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matriz, annot=True, fmt='d', cmap='Blues',
        xticklabels=nombres_clases, yticklabels=nombres_clases,
        linewidths=0.5
    )
    plt.title('Matriz de Confusion Global (15 Rodamientos LOBO)', fontweight='bold', fontsize=13)
    plt.ylabel('Etiqueta Real', fontsize=11)
    plt.xlabel('Prediccion del Modelo', fontsize=11)
    plt.tight_layout()
    plt.savefig('matriz_confusion.png', dpi=150, bbox_inches='tight')
    plt.show()

    # --- Evidencia 3: Curva de convergencia (loss promedio) ---
    loss_promedio = np.mean(historial_loss, axis=0)
    loss_std      = np.std(historial_loss,  axis=0)
    epocas_eje    = range(1, EPOCAS + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epocas_eje, loss_promedio, color='#2196F3', linewidth=2, label='Loss promedio')
    plt.fill_between(
        epocas_eje,
        loss_promedio - loss_std,
        loss_promedio + loss_std,
        alpha=0.2, color='#2196F3', label='Desv. std (15 folds)'
    )
    plt.title('Convergencia del Entrenamiento (Promedio de 15 Folds)', fontweight='bold', fontsize=13)
    plt.xlabel('Epoca', fontsize=11)
    plt.ylabel('Perdida (Cross-Entropy Loss)', fontsize=11)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('curva_convergencia.png', dpi=150, bbox_inches='tight')
    plt.show()

    generar_tabla_comparativa(acc_train_final, acc_test_final, resultados_test)


# =============================================================================
# TABLA COMPARATIVA + GRÁFICO POR RODAMIENTO
# =============================================================================
def generar_tabla_comparativa(acc_cnn_train, acc_cnn_test, resultados_por_rod):

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9))

    # --- Tabla superior ---
    filas = [
        {'Modelo': 'Decision Tree (Paper)',          'Features': 'Time/Freq (Handcrafted)', 'Acc. Train (%)': '100.0', 'Acc. LOBO Test (%)': '47.1'},
        {'Modelo': 'SVM (Paper)',                    'Features': 'Time/Freq (Handcrafted)', 'Acc. Train (%)': '88.3',  'Acc. LOBO Test (%)': '68.5'},
        {'Modelo': 'k-NN (Paper)',                   'Features': 'Time/Freq (Handcrafted)', 'Acc. Train (%)': '79.1',  'Acc. LOBO Test (%)': '55.6'},
        {'Modelo': 'Random Forest (Paper)',          'Features': 'Time/Freq (Handcrafted)', 'Acc. Train (%)': '100.0', 'Acc. LOBO Test (%)': '66.1'},
        {'Modelo': 'CNN 1D + FFT Log (Corregida)',   'Features': 'Frequency Spectrum (Log)','Acc. Train (%)': f"{acc_cnn_train:.1f}", 'Acc. LOBO Test (%)': f"{acc_cnn_test:.1f}"},
    ]
    df = pd.DataFrame(filas)

    ax1.axis('off')
    tbl = ax1.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2.2)

    for j in range(len(df.columns)):
        tbl[0, j].set_facecolor('#263238')
        tbl[0, j].set_text_props(color='white', weight='bold')

    for i in range(1, len(df) + 1):
        color = '#E8F5E9' if i == 5 else ('#ECEFF1' if i % 2 == 0 else 'white')
        for j in range(len(df.columns)):
            tbl[i, j].set_facecolor(color)
            if i == 5:
                tbl[i, j].set_text_props(weight='bold', color='#1B5E20')

    ax1.set_title('Tabla Comparativa: Modelos Clásicos vs CNN 1D + FFT Log (Corregida)',
                  fontsize=13, fontweight='bold', pad=10)

    # --- Gráfico inferior: Accuracy por rodamiento ---
    colores = ['#4CAF50' if r >= 68.5 else '#FF7043' if r < 33.3 else '#FFA726'
               for r in resultados_por_rod]

    ax2.bar(RODAMIENTOS_TABLA_10, resultados_por_rod, color=colores, edgecolor='white', linewidth=0.8)
    ax2.axhline(y=68.5, color='#1565C0', linestyle='--', linewidth=1.5, label='SVM Paper (68.5%)')
    ax2.axhline(y=33.3, color='#B71C1C', linestyle=':',  linewidth=1.5, label='Azar (33.3%)')
    ax2.set_ylim(0, 105)
    ax2.set_ylabel('Accuracy LOBO (%)', fontsize=11)
    ax2.set_title('Accuracy por Rodamiento (Verde ≥ SVM | Naranja < SVM | Rojo < Azar)', fontsize=11)
    ax2.legend(fontsize=10)
    ax2.tick_params(axis='x', rotation=35)

    for i, (rod, val) in enumerate(zip(RODAMIENTOS_TABLA_10, resultados_por_rod)):
        ax2.text(i, val + 1.5, f'{val:.0f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout(pad=3.0)
    plt.savefig('resultados_fft_corregido.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Gráfico guardado como 'resultados_fft_corregido.png'")


# =============================================================================
# EJECUCIÓN
# =============================================================================
if __name__ == '__main__':
    ejecutar_lobo_oficial()
