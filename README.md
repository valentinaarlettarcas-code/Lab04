# Laboratoria 4 - Análisis Exploratorio de Datos (EDA)
## Dataset: IMDb movies Extensive dataset
Proyecto grupal de la asignatura Big Data. Análisis exploratorio reproducible sobre un dataset de peliculas IMDb con más de 85.000 registros, implementado en Pyhton con pandas, NumPy, Matplotlib, Seaborn y SciPy.


## Version de Python
- **Python 3.12.x o superior**: requerido por el uso de `type hints int | None ` y `dict[str, ...]` en alisis_exporatorio.py
- Verificar con: ` python --version`

## Instalación de Dependencias
1.  Crear y activar un entorno virtual (Opcional pero recomendado):
```
# creación de entorno
python3 -m venv venv

# activación
source venv/bin/activate        # Linux/MacOS/WSL
venv\Scripts\activate.bat       # Windows CMD
.\venv\Scripts\Activate.ps1     # Windows PowerShell
```
2. Instalar las dependencias:

` pip install -r requirements.txt`

Las versiones de las librerias se encuentran en `requitements.txt` para garantizar la reproducibilidad.

## Dataset

El dataset utilizado en este proyecto corresponde al **IMDb movies extensive dataset**. 
- **Título:** imdb-extensive-dataset
- **Autor:** Simhyunsu
- **URL de descarga:** https://www.kaggle.com/datasets/simhyunsu/imdbextensivedataset/data
- **Fecha de descarga:** 18 de octubre de 2026
- **Licencia:** [POR CONFIRMAR ]
- **Tamaño:** 85.855 registros
- **Ubicación esperada:** colocar el archivo `IMDb movies.csv` dentro de la
  carpeta `dataset/` antes de ejecutar el script

## Regeneración del dataset procesado
El dataset limpio se recontruye integramente ejecutando `analisis_exploratorio.py` (no en el archivo original), que parte de `dataset/IMDb movies.csv` y se aplican las transformaciones documentadas en el informe (Sección 4: Diagnóstico y decisiones de limpieza). Cada paso queda registrado en `resultados/tablas/registro_limpieza.csv`con cantidades antes y después, por lo que el proceso es trazable.
## Estructura del proyecto
``` 
Lab04
├── README.md
├── analisis_exploratorio.py        # Script principal (Ejecuta todo el EDA)
├── dataset
│   └── IMDb movies.csv             # Dataset original
├── requirements.txt                # Dependencias y versiones
├── resultados                      # Generados en la ejecución
│   ├── figuras                     # Gráficos del análisis
│   └── tablas                      # Diagnóstico, estadísticas y limpieza
```
## Ejecución
Desde la carpeta raíz del proyecto:
`analisis_exploratorio.py`
Para usar una ruta distinta al dataset: 
`python analisis_exploratorio.py --ruta "ruta/a/otro_archivo.csv"`

### Reproducibilidad
- **Semilla**: el script fija SEMILLA = 42 al inicio (`np.random.seed(42)`) para garantizar reproducibilidad. Se analiza el conjunto completo..
- **Rutas relativas:** todo el script trabaja con rutas relativas a la carpetadonde vive `analisis_exploratorio.py` (vía `Path(__file__).resolve().parent`).No hay rutas absolutas ni personales en el código.
- **Carpetas de salida:** `resultados/tablas/` y `resultados/figuras/` se crean automáticamente al arranque. Si ya existen y contienen archivos de una ejecución anterior, los CSV/PNG se sobrescriben con los nuevos resultados.
- **Datos personales:** las columnas director, writer y actors se excluyen del análisis y de todos los entregables por indicación del enunciado.

### Etapas del Script
1. **Carga e inspección**: lee el CSV original del dataset, detectando codificacion y separadores, valida el mínimo de 50.000 registros e informa dimenciones, tipos y primeras observaciones.

2. **Diagnóstico de calidad**: cuantifica faltantes, duplicados (exactos vs. repeticiones válidas), tipos inconsistentes, valores fuera de rango plausible y monedas mezcladas. Genera 6 tablas CSV (`diagnostico_*.csv`).

3. **Limpieza y preparación**: trata los problemas detectados con decisiones justificadas y deja registro antes/después de cada paso en `registro_limpieza.csv`. Genera variables derivadas (decada, log_votes,genero_principal, pais_principal, generos).

4. **Estadística descriptiva**:
- Variables numéricas: n, media, mediana, desviación, cuartiles, IQR y asimetría; 
- Categóricas: frecuencias y porcentajes.

5. **Comparación por grupos y correlación**: agregaciones por género, país y década; matriz de correlación de Spearman y Pearson, con comparación por grupos como control de confusión.

6. **Visualización**: 5 gráficos correspondiente a distribución, valores extremos, dispersión por grupos, recuencias, relación y

7. **Exportación**: todas las tablas generadas se agregan a `resultados/tablas/` en CSV y todas las figuras a `resultados/figuras/` en PNG. Al final imprime un resumen con el conteo de registros de entrada, registros finales y archivos generados.


## Salidas

 ### `resultados/tablas/` (archivos CSV)

| Origen | Archivos |
|---|---|
| Diagnóstico  | `diagnostico_resumen.csv`, `diagnostico_faltantes.csv`, `diagnostico_duplicados.csv`, `diagnostico_tipos.csv`, `diagnostico_rangos.csv`, `diagnostico_monedas.csv` |
| Limpieza | `registro_limpieza.csv`, `sesgo_genero_principal.csv` |
| Estadística | `descriptiva_numericas.csv`, `frecuencia_genero.csv`, `frecuencia_pais.csv`, `frecuencia_decada.csv`, `grupo_genero_avg_vote.csv`, `grupo_genero_duracion.csv`, `grupo_pais_avg_vote.csv`, `evolucion_decada.csv`, `correlacion_spearman.csv`, `correlacion_pearson.csv`, `pearson_vs_spearman.csv`, `correlacion_votos_nota_por_genero.csv` |

### `resultados/figuras/` (archivos PNG)

1. `1_distribucion_nota.png` — Distribución de la calificación promedio
2. `2_valores_extremos_duracion.png` — Valores extremos de la duración
3. `3_dispersion_grupos_genero.png` — Dispersión por grupos (género)
4. `4_frecuencias_paises.png` — Frecuencias por país (Top 10)
5. `5_relacion_duracion_nota.png` — Relación entre duración y nota
6. `6_pregunta_equipo_evolucion_duracion.png` — Evolución de la duración mediana por década


## Equipo

|Integrante|	Responsabilidad principal|
|--|--|
|Antonio Gómez|	Búsqueda y descarga del dataset; contexto, objetivo, 4 preguntas y diccionario de datos|
|Valentina Arcas|	Carga e inspección, diagnóstico de calidad, sección Metodología del informe|
|Benjamín Lagos|	Limpieza y preparación, registro antes/después de cada transformación|
|Iván Herrera|	Estadística descriptiva, comparación por grupos, correlación Spearman/Pearson|
|Vianca Viveros	|6 figuras PNG con Matplotlib/Seaborn, interpretación visual en el informe|
|Carloth Salgado|	Exportación, rutas, requirements.txt, README.md, respuesta de las preguntas, cierre del informe y ZIP final|