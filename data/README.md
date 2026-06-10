# Data

Los archivos `.npy` no están versionados por su tamaño (~4 GB).

## Descarga del dataset original

1. Ir a https://mb.uni-paderborn.de/kat/forschung/datacenter/bearing-datacenter/
2. Descargar los archivos de los rodamientos listados en `docs/`
3. Filtrar: 1500 RPM, excluir casos KB (mixed fault)

## Generar los archivos .npy

Una vez descargado el dataset raw, ejecutar en orden:

# 1. Ventaneo y fusión de canales
python preprocessing/ventaneo.py

# 2. Extracción de features (272 descriptores)
python features/feature_extraction.py

## Archivos esperados en data/
- X_feat_crudos.npy       # (N, 272) features
- Y_labels_crudos.npy     # (N,) etiquetas
- Y_codigos_crudos.npy    # (N,) códigos de rodamiento
- feature_names_crudos.npy