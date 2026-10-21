"""
analisis_exploratorio.py — Laboratorio 4 Big Data: EDA del dataset "IMDb movies extensive dataset".

Ejecución (desde la carpeta del proyecto):
    python analisis_exploratorio.py
    python analisis_exploratorio.py --ruta "dataset/IMDb movies.csv"

Etapas del script (cada integrante completa su sección):
    1-2. Carga e inspección / Diagnóstico de calidad ...... Valentina
    3-4. Limpieza, preparación y registro de limpieza ..... Benjamin
    5-6. Estadística descriptiva, grupos y correlación .... Iván
    7.   Visualizaciones .................................. Vianca
    8.   Exportación, rutas y carpetas .................... Charloth
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# 0. CONFIGURACIÓN GENERAL (rutas relativas a la carpeta del script)
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent
RUTA_DATOS_POR_DEFECTO = BASE_DIR / "dataset" / "IMDb movies.csv"
DIR_TABLAS = BASE_DIR / "resultados" / "tablas"
DIR_FIGURAS = BASE_DIR / "resultados" / "figuras"
SEMILLA = 42
MIN_REGISTROS = 50_000

VARIABLES_NUMERICAS = [
    "avg_vote", "votes", "duration", "year",
    "reviews_from_users", "reviews_from_critics",
]
TOP_CATEGORIAS = 10      # máximo de categorías por tabla/grupo (legibilidad)
MIN_OBS_GRUPO = 30       # grupos con menos películas no se comparan


# =============================================================================
# 8. EXPORTACIÓN (Charloth reemplaza/amplía esta función)
# =============================================================================
def guardar_tabla(df: pd.DataFrame, nombre: str) -> None:
    """Exporta una tabla a resultados/tablas/<nombre>.csv."""
    DIR_TABLAS.mkdir(parents=True, exist_ok=True)
    df.to_csv(DIR_TABLAS / f"{nombre}.csv", index=True, encoding="utf-8")
    print(f"  [tabla] {nombre}.csv")


# =============================================================================
# 1-2. CARGA, INSPECCIÓN Y DIAGNÓSTICO (Valentina reemplaza esta versión mínima)
# =============================================================================
def cargar_datos(ruta: Path) -> pd.DataFrame:
    """Lectura mínima provisoria. Valentina agrega inspección y diagnóstico."""
    if not ruta.exists():
        sys.exit(f"ERROR: no se encontró el archivo de datos en '{ruta}'.")
    df = pd.read_csv(ruta, low_memory=False, encoding="utf-8")
    print(f"Registros leídos: {len(df):,} | Columnas: {df.shape[1]}")
    if len(df) < MIN_REGISTROS:
        sys.exit(f"ERROR: el dataset tiene {len(df):,} registros (< {MIN_REGISTROS:,}).")
    return df


# =============================================================================
# 3-4. LIMPIEZA Y PREPARACIÓN (Benjamin amplía; NO cambiar los nombres de las
#      columnas derivadas porque las secciones 5-7 dependen de ellas)
# =============================================================================
def preparar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Versión mínima: convierte `year` y crea las variables derivadas que usa
    el análisis. Benjamin añade aquí el tratamiento de nulos, extremos y el
    registro de limpieza (antes/después de cada paso).
    """
    df = df.copy()
    # `year` llega como texto por una fila con valor 'TV Movie 2019'
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    # Variables derivadas (documentadas en el diccionario de datos)
    df["decada"] = (df["year"] // 10) * 10                      # década de estreno
    df["log_votes"] = np.log10(df["votes"])                      # votos en escala log10
    df["genero_principal"] = df["genre"].str.split(",").str[0].str.strip()
    df["pais_principal"] = df["country"].str.split(",").str[0].str.strip()
    return df


# =============================================================================
# 5-6. ESTADÍSTICA DESCRIPTIVA, COMPARACIÓN POR GRUPOS Y CORRELACIÓN (Iván)
# =============================================================================
# ---------------------------------------------------------------------------
# 5.1 Estadística descriptiva de variables numéricas
# ---------------------------------------------------------------------------
def estadistica_descriptiva_numerica(df: pd.DataFrame, columnas=VARIABLES_NUMERICAS) -> pd.DataFrame:
    """n válidos, faltantes, media, mediana, desviación, mín, Q1, Q3, máx, IQR y asimetría."""
    filas = {}
    for col in columnas:
        s = pd.to_numeric(df[col], errors="coerce")
        q1, q3 = s.quantile([0.25, 0.75])
        filas[col] = {
            "n_validos": int(s.notna().sum()),
            "n_faltantes": int(s.isna().sum()),
            "media": s.mean(),
            "mediana": s.median(),
            "desv_estandar": s.std(),          # ddof=1 (muestral)
            "minimo": s.min(),
            "q1": q1,
            "q3": q3,
            "maximo": s.max(),
            "iqr": q3 - q1,
            "asimetria": s.skew(),             # >1 ⇒ cola larga a la derecha
        }
    tabla = pd.DataFrame(filas).T.round(3)
    tabla.index.name = "variable"
    return tabla


# ---------------------------------------------------------------------------
# 5.2 Frecuencias y porcentajes de variables categóricas
# ---------------------------------------------------------------------------
def frecuencias_categoricas(df: pd.DataFrame, columna: str, top: int | None = TOP_CATEGORIAS,
                            orden_natural: bool = False) -> pd.DataFrame:
    """Frecuencia absoluta, % y % acumulado. Agrupa el resto en 'Otros' si se pide top.
    orden_natural=True ordena por la categoría (útil para décadas) en vez de por frecuencia."""
    conteo = df[columna].value_counts(dropna=False)
    if orden_natural:
        conteo = conteo.sort_index()
    if top is not None and len(conteo) > top:
        otros = conteo.iloc[top:].sum()
        conteo = conteo.iloc[:top]
        conteo.loc["Otros"] = otros
    tabla = pd.DataFrame({
        "frecuencia": conteo,
        "porcentaje": (conteo / conteo.sum() * 100).round(2),
    })
    tabla["porcentaje_acumulado"] = tabla["porcentaje"].cumsum().round(2)
    tabla.index.name = columna
    return tabla


# ---------------------------------------------------------------------------
# 5.3 Comparación / agregación por grupos
# ---------------------------------------------------------------------------
def comparar_por_grupo(df: pd.DataFrame, grupo: str, variable: str,
                       top: int | None = TOP_CATEGORIAS) -> pd.DataFrame:
    """Resumen de `variable` dentro de cada categoría de `grupo` (ordenado por mediana)."""
    datos = df[[grupo, variable]].dropna()
    if top is not None:
        principales = datos[grupo].value_counts().head(top).index
        datos = datos[datos[grupo].isin(principales)]
    tabla = (datos.groupby(grupo)[variable]
             .agg(n="count", media="mean", mediana="median", desv_estandar="std",
                  q1=lambda s: s.quantile(0.25), q3=lambda s: s.quantile(0.75),
                  minimo="min", maximo="max"))
    tabla = tabla[tabla["n"] >= MIN_OBS_GRUPO]
    tabla = tabla.sort_values("mediana", ascending=False).round(3)
    return tabla


def evolucion_por_decada(df: pd.DataFrame) -> pd.DataFrame:
    """Agregación temporal: cantidad de películas y métricas centrales por década."""
    tabla = (df.groupby("decada")
             .agg(n_peliculas=("avg_vote", "size"),
                  avg_vote_media=("avg_vote", "mean"),
                  avg_vote_mediana=("avg_vote", "median"),
                  duracion_mediana=("duration", "median"),
                  votos_mediana=("votes", "median"))
             .round(3))
    tabla["pct_peliculas"] = (tabla["n_peliculas"] / tabla["n_peliculas"].sum() * 100).round(2)
    return tabla


# ---------------------------------------------------------------------------
# 6. Relaciones entre variables (correlación)
# ---------------------------------------------------------------------------
def matriz_correlacion(df: pd.DataFrame, columnas, metodo: str = "spearman") -> pd.DataFrame:
    """
    Matriz de correlación por pares (pairwise: cada par usa sus filas sin NaN).
    Se usa Spearman por defecto: votes y reviews tienen asimetría extrema y
    outliers, y Spearman mide asociación monótona sobre rangos, sin suponer
    linealidad ni normalidad. NO se incluye imdb_title_id (es un identificador).
    """
    return df[columnas].corr(method=metodo).round(3)


def comparar_pearson_spearman(df: pd.DataFrame, pares) -> pd.DataFrame:
    """Para cada par de variables informa n, Pearson y Spearman lado a lado."""
    filas = []
    for x, y in pares:
        d = df[[x, y]].dropna()
        filas.append({
            "variable_x": x, "variable_y": y, "n": len(d),
            "pearson": round(d[x].corr(d[y], method="pearson"), 3),
            "spearman": round(d[x].corr(d[y], method="spearman"), 3),
        })
    return pd.DataFrame(filas).set_index(["variable_x", "variable_y"])


def correlacion_por_grupo(df: pd.DataFrame, grupo: str, x: str, y: str,
                          top: int = TOP_CATEGORIAS) -> pd.DataFrame:
    """¿La relación x–y se mantiene dentro de cada grupo? (control simple de confusión)."""
    principales = df[grupo].value_counts().head(top).index
    filas = {}
    for g in principales:
        d = df.loc[df[grupo] == g, [x, y]].dropna()
        if len(d) >= MIN_OBS_GRUPO:
            filas[g] = {"n": len(d), "spearman": round(d[x].corr(d[y], method="spearman"), 3)}
    tabla = pd.DataFrame(filas).T
    tabla.index.name = grupo
    return tabla.sort_values("spearman", ascending=False)


# ---------------------------------------------------------------------------
# Orquestador de la sección (lo llama main() del script principal)
# ---------------------------------------------------------------------------
def ejecutar_analisis_estadistico(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    print("\n=== 5. ESTADÍSTICA DESCRIPTIVA ===")
    resultados = {
        "descriptiva_numericas": estadistica_descriptiva_numerica(df),
        "frecuencia_genero": frecuencias_categoricas(df, "genero_principal"),
        "frecuencia_pais": frecuencias_categoricas(df, "pais_principal"),
        "frecuencia_decada": frecuencias_categoricas(df, "decada", top=None, orden_natural=True),
        "grupo_genero_avg_vote": comparar_por_grupo(df, "genero_principal", "avg_vote"),
        "grupo_genero_duracion": comparar_por_grupo(df, "genero_principal", "duration"),
        "grupo_pais_avg_vote": comparar_por_grupo(df, "pais_principal", "avg_vote"),
        "evolucion_decada": evolucion_por_decada(df),
    }

    print("\n=== 6. RELACIONES ENTRE VARIABLES ===")
    cols_corr = ["avg_vote", "votes", "duration", "year",
                 "reviews_from_users", "reviews_from_critics"]
    resultados["correlacion_spearman"] = matriz_correlacion(df, cols_corr, "spearman")
    resultados["correlacion_pearson"] = matriz_correlacion(df, cols_corr, "pearson")
    resultados["pearson_vs_spearman"] = comparar_pearson_spearman(df, [
        ("votes", "avg_vote"), ("log_votes", "avg_vote"),
        ("duration", "avg_vote"), ("year", "avg_vote"),
        ("reviews_from_users", "reviews_from_critics"),
    ])
    resultados["correlacion_votos_nota_por_genero"] = correlacion_por_grupo(
        df, "genero_principal", "log_votes", "avg_vote")

    for nombre, tabla in resultados.items():
        guardar_tabla(tabla, nombre)
    return resultados


# =============================================================================
# 7. VISUALIZACIONES (Vianca)
# =============================================================================
def generar_graficos(df: pd.DataFrame, resultados: dict) -> None:
    """Pendiente: 5 gráficos obligatorios + gráfico de la pregunta del equipo."""
    pass


# =============================================================================
# PROGRAMA PRINCIPAL
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="EDA IMDb movies — Laboratorio 4")
    parser.add_argument("--ruta", type=Path, default=RUTA_DATOS_POR_DEFECTO,
                        help="ruta al CSV de entrada (por defecto dataset/IMDb movies.csv)")
    args = parser.parse_args()
    np.random.seed(SEMILLA)

    df_original = cargar_datos(args.ruta)
    df_limpio = preparar_datos(df_original)
    resultados = ejecutar_analisis_estadistico(df_limpio)
    generar_graficos(df_limpio, resultados)
    print("\nProceso finalizado correctamente.")


if __name__ == "__main__":
    main()
