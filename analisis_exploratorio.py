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
import csv
from datetime import date
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
def inicializar_carpetas() -> None:
    """Crea las carpetas de salida (idempotente)."""
    DIR_TABLAS.mkdir(parents=True, exist_ok=True)
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    print(f"Carpetas de salida listas:\n  {DIR_TABLAS}\n  {DIR_FIGURAS}")

def guardar_tabla(df: pd.DataFrame, nombre: str) -> None:
    """Exporta una tabla a resultados/tablas/<nombre>.csv."""
    DIR_TABLAS.mkdir(parents=True, exist_ok=True)
    df.to_csv(DIR_TABLAS / f"{nombre}.csv", index=True, encoding="utf-8")
    print(f"  [tabla] {nombre}.csv")

def resumen_ejecucion(df_limpio: pd.DataFrame = None, n_entrada: int = None) -> None:
    """Imprime un resumen de archivos generados (Anexo - Ejecución)."""
    tablas = sorted(DIR_TABLAS.glob("*.csv"))
    figuras = sorted(DIR_FIGURAS.glob("*.png"))
    
    print("=== RESUMEN DE EJECUCIÓN ===")
    print("="*30)
    if df_limpio is not None:
        n_final= len(df_limpio)
        if n_entrada is not None:
            print(f"Registros de entrada: {n_entrada}")
            eliminados = n_entrada - n_final
            print(f"Registros eliminados: {eliminados}")
            print(f"Registros finales: {n_final}")
            print("-"*30)

        else:
            print(f"Registros de entrada: no informado")

    print(f"Tablas generadas: {len(tablas)}")
    for f in tablas:
        print(f"  - {f.name}")
    print("-"*30)
    print(f"Figuras generadas: {len(figuras)}")
    for f in figuras:
        print(f"  - {f.name}")
    print("="*30)

# =============================================================================
# 1-2. CARGA, INSPECCIÓN Y DIAGNÓSTICO DE CALIDAD (Valentina)
# =============================================================================
# Requiere estos imports al inicio del archivo (junto a los existentes):
#     import csv
#     from datetime import date

# Columnas que el análisis necesita sí o sí (si falta alguna, se avisa y se detiene)
COLUMNAS_REQUERIDAS = [
    "imdb_title_id", "title", "year", "date_published", "genre", "duration",
    "country", "avg_vote", "votes", "budget", "usa_gross_income",
    "worlwide_gross_income", "reviews_from_users", "reviews_from_critics",
]
# Nombres de personas: el enunciado pide no usar datos personales, así que se
# excluyen del análisis y de todos los entregables (después del diagnóstico).
COLUMNAS_PERSONALES = ["director", "writer", "actors"]
COLUMNAS_NUMERICAS_ESPERADAS = [
    "year", "duration", "avg_vote", "votes", "metascore",
    "reviews_from_users", "reviews_from_critics",
]
# Vienen como texto con prefijo de moneda (ej. "$ 2250", "INR 1000"). OJO: la
# columna se llama "worlwide_gross_income" (sin la "d") en el dataset original.
COLUMNAS_MONETARIAS = ["budget", "usa_gross_income", "worlwide_gross_income"]
FORMATO_FECHA = "%Y-%m-%d"
# variable: (mínimo plausible, máximo plausible, criterio)
RANGOS_PLAUSIBLES = {
    "year": (1870, date.today().year, "el cine comienza en la década de 1890; no hay estrenos futuros"),
    "avg_vote": (1, 10, "escala de votación de IMDb"),
    "votes": (0, None, "no puede haber votos negativos"),
    "duration": (1, 1000, "minutos; fuera de rango puede ser real (maratones, TV), se revisa con cautela"),
    "metascore": (0, 100, "escala del Metascore"),
    "reviews_from_users": (0, None, "conteo, no puede ser negativo"),
    "reviews_from_critics": (0, None, "conteo, no puede ser negativo"),
}


def _detectar_separador(ruta: Path, encoding: str) -> str:
    """Elige el separador más frecuente en la línea de encabezados."""
    with open(ruta, "r", encoding=encoding, errors="replace", newline="") as f:
        encabezado = f.readline()
    conteo = {sep: encabezado.count(sep) for sep in [",", ";", "\t", "|"]}
    return max(conteo, key=conteo.get)


def _leer_csv(ruta: Path) -> pd.DataFrame:
    """Lee el CSV detectando codificación y separador, con mensajes de error claros."""
    if not ruta.exists():
        sys.exit(f"ERROR: no se encontró el archivo de datos en '{ruta}'. "
                 "Revisa la ruta o pásala con --ruta.")
    if ruta.suffix.lower() != ".csv":
        print(f"ADVERTENCIA: '{ruta.name}' no tiene extensión .csv, se intenta leer igual.")

    df = None
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            sep = _detectar_separador(ruta, encoding)
            df = pd.read_csv(ruta, sep=sep, encoding=encoding, low_memory=False)
            nombre_sep = {"\t": "tabulación"}.get(sep, sep)
            print(f"Archivo leído con codificación '{encoding}' y separador '{nombre_sep}'.")
            break
        except UnicodeDecodeError:
            print(f"  La codificación '{encoding}' falló, se prueba la siguiente.")
        except pd.errors.EmptyDataError:
            sys.exit(f"ERROR: el archivo '{ruta.name}' está vacío.")
        except pd.errors.ParserError as e:
            sys.exit(f"ERROR: el archivo '{ruta.name}' tiene un formato CSV inválido: {e}")
    if df is None:
        sys.exit(f"ERROR: no se pudo decodificar '{ruta.name}' con ninguna codificación probada.")
    if df.shape[1] <= 1:
        sys.exit("ERROR: el archivo se leyó con una sola columna; "
                 "probablemente el separador no es el esperado.")
    return df


def inspeccionar_datos(df: pd.DataFrame) -> None:
    """Informa dimensiones, columnas, tipos y primeras filas; valida el mínimo de registros."""
    print("\n=== 1. CARGA E INSPECCIÓN ===")
    print(f"Registros leídos: {len(df):,} | Columnas: {df.shape[1]}")
    print("\nColumnas y tipos de dato (tal como los infiere pandas):")
    print(df.dtypes.astype(str).to_string())

    # Las columnas con nombres de personas no se muestran en pantalla
    vista = [c for c in ["imdb_title_id", "title", "year", "genre", "duration",
                         "country", "avg_vote", "votes", "budget"] if c in df.columns]
    print("\nPrimeras observaciones (sin columnas con nombres de personas):")
    print(df[vista].head(5).to_string(max_colwidth=28))

    faltan = [c for c in COLUMNAS_REQUERIDAS if c not in df.columns]
    if faltan:
        sys.exit(f"ERROR: faltan columnas necesarias en el archivo: {faltan}")
    # El mínimo se valida con los datos tal como llegaron, antes de cualquier limpieza
    if len(df) < MIN_REGISTROS:
        sys.exit(f"ERROR: el dataset tiene {len(df):,} registros (mínimo exigido: {MIN_REGISTROS:,}).")
    print(f"\nOK: {len(df):,} registros de entrada (mínimo {MIN_REGISTROS:,}).")


def _tabla_faltantes(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    tabla = pd.DataFrame({
        "faltantes": df.isna().sum(),
        "porcentaje": (df.isna().sum() / n * 100).round(2),
    }).sort_values("faltantes", ascending=False)
    tabla.index.name = "columna"
    return tabla


def _tabla_duplicados(df: pd.DataFrame) -> pd.DataFrame:
    """Separa duplicados exactos, ID repetido y títulos repetidos con ID distinto."""
    exactos = int(df.duplicated(keep="first").sum())
    id_repetido = int(df.duplicated(subset=["imdb_title_id"], keep="first").sum())
    ids_por_titulo = df.groupby(["title", "year"], dropna=False)["imdb_title_id"].nunique()
    grupos_repetidos = ids_por_titulo[ids_por_titulo > 1]
    filas = [
        {"tipo": "Duplicados exactos (todas las columnas iguales)", "cantidad": exactos,
         "interpretacion": "Error de registro: la misma fila repetida.",
         "decision": "Eliminar" if exactos else "Sin problema detectado"},
        {"tipo": "imdb_title_id repetido", "cantidad": id_repetido,
         "interpretacion": "El identificador debería ser único por película.",
         "decision": "Revisar y conservar una fila por ID" if id_repetido else "Sin problema detectado"},
        {"tipo": "Mismo título y año con ID distinto (filas involucradas)",
         "cantidad": int(grupos_repetidos.sum()),
         "interpretacion": "Repetición válida: son películas distintas con el mismo nombre.",
         "decision": "Conservar (no es duplicado)"},
    ]
    tabla = pd.DataFrame(filas).set_index("tipo")
    return tabla


def _tabla_tipos(df: pd.DataFrame) -> pd.DataFrame:
    """Tipos de dato y valores que no se pueden convertir al tipo esperado."""
    filas = {}
    for col in df.columns:
        filas[col] = {"tipo_pandas": str(df[col].dtype), "no_convertibles": 0, "ejemplos": ""}
    for col in COLUMNAS_NUMERICAS_ESPERADAS:
        if col in df.columns:
            conv = pd.to_numeric(df[col], errors="coerce")
            malos = df[col][conv.isna() & df[col].notna()]
            filas[col]["no_convertibles"] = int(len(malos))
            filas[col]["ejemplos"] = " | ".join(map(str, malos.unique()[:3]))
    fechas = pd.to_datetime(df["date_published"], format=FORMATO_FECHA, errors="coerce")
    malas = df["date_published"][fechas.isna() & df["date_published"].notna()]
    filas["date_published"]["no_convertibles"] = int(len(malas))
    filas["date_published"]["ejemplos"] = " | ".join(map(str, malas.unique()[:3]))
    tabla = pd.DataFrame(filas).T
    tabla.index.name = "columna"
    return tabla


def _tabla_rangos(df: pd.DataFrame) -> pd.DataFrame:
    """Valores fuera de rangos plausibles (se cuentan, NO se eliminan aquí)."""
    filas = {}
    n = len(df)
    for col, (lo, hi, criterio) in RANGOS_PLAUSIBLES.items():
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        fuera = pd.Series(False, index=s.index)
        if lo is not None:
            fuera |= s < lo
        if hi is not None:
            fuera |= s > hi
        filas[col] = {
            "rango_plausible": f"[{lo}, {hi if hi is not None else '∞'}]",
            "n_validos": int(s.notna().sum()),
            "minimo_obs": s.min(), "maximo_obs": s.max(),
            "fuera_de_rango": int(fuera.sum()),
            "porcentaje": round(fuera.sum() / n * 100, 3),
            "criterio": criterio,
        }
    # Montos en cero: en este dataset suelen representar "dato desconocido"
    for col in COLUMNAS_MONETARIAS:
        digitos = pd.to_numeric(df[col].dropna().astype(str).str.replace(r"\D", "", regex=True),
                                errors="coerce")
        ceros = int((digitos == 0).sum())
        filas[f"{col} == 0"] = {
            "rango_plausible": "> 0", "n_validos": int(len(digitos)),
            "minimo_obs": digitos.min(), "maximo_obs": digitos.max(),
            "fuera_de_rango": ceros, "porcentaje": round(ceros / n * 100, 3),
            "criterio": "un monto 0 probablemente significa dato desconocido",
        }
    tabla = pd.DataFrame(filas).T
    tabla.index.name = "variable"
    return tabla


def _tabla_monedas(df: pd.DataFrame) -> pd.DataFrame:
    """Prefijo de moneda de las columnas de dinero (no se pueden sumar monedas distintas)."""
    partes = []
    for col in COLUMNAS_MONETARIAS:
        s = df[col].dropna().astype(str).str.strip()
        prefijo = s.str.extract(r"^([^\d\s.,-]+)")[0].fillna("(sin prefijo)")
        conteo = prefijo.value_counts().head(8)
        t = pd.DataFrame({"columna": col, "moneda": conteo.index, "registros": conteo.values})
        t["porcentaje_de_no_nulos"] = (t["registros"] / len(s) * 100).round(2)
        partes.append(t)
    return pd.concat(partes).set_index(["columna", "moneda"])


def diagnosticar_calidad(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Diagnóstico: faltantes, duplicados, tipos inconsistentes y rangos. Solo informa, no modifica."""
    print("\n=== 2. DIAGNÓSTICO DE CALIDAD ===")
    faltantes = _tabla_faltantes(df)
    duplicados = _tabla_duplicados(df)
    tipos = _tabla_tipos(df)
    rangos = _tabla_rangos(df)
    monedas = _tabla_monedas(df)
    resumen = pd.DataFrame({
        "valor": [len(df), df.shape[1], int(df.isna().any(axis=1).sum()),
                  ", ".join(c for c in COLUMNAS_PERSONALES if c in df.columns)]},
        index=["registros_de_entrada", "columnas", "filas_con_algun_faltante",
               "columnas_excluidas_por_datos_personales"])
    resumen.index.name = "indicador"

    print(f"Columnas con más faltantes:\n{faltantes.head(5).to_string()}")
    print(f"\nDuplicados:\n{duplicados['cantidad'].to_string()}")
    malos = tipos[tipos["no_convertibles"] > 0]
    print(f"\nColumnas con valores no convertibles:\n"
          f"{malos[['no_convertibles', 'ejemplos']].to_string() if len(malos) else 'ninguna'}")
    print(f"\nValores fuera de rango plausible:\n{rangos['fuera_de_rango'].to_string()}")

    return {
        "diagnostico_resumen": resumen,
        "diagnostico_faltantes": faltantes,
        "diagnostico_duplicados": duplicados,
        "diagnostico_tipos": tipos,
        "diagnostico_rangos": rangos,
        "diagnostico_monedas": monedas,
    }


def excluir_columnas_personales(df: pd.DataFrame) -> pd.DataFrame:
    """Quita nombres de personas del dataset que sigue en el pipeline y de los entregables."""
    presentes = [c for c in COLUMNAS_PERSONALES if c in df.columns]
    print(f"\nColumnas excluidas por contener nombres de personas: {presentes}")
    return df.drop(columns=presentes)


def cargar_datos(ruta: Path) -> pd.DataFrame:
    """Carga, inspecciona y diagnostica. Devuelve los datos SIN limpiar (salvo columnas personales)."""
    df = _leer_csv(ruta)
    n_entrada = len(df)
    inspeccionar_datos(df)
    for nombre, tabla in diagnosticar_calidad(df).items():
        guardar_tabla(tabla, nombre)
    df = excluir_columnas_personales(df)
    # Benjamin puede usar este valor para el registro de limpieza (antes/después)
    df.attrs["n_registros_entrada"] = n_entrada
    return df


# =============================================================================
# 3-4. LIMPIEZA Y PREPARACIÓN (Benjamin amplía; NO cambiar los nombres de las
#      columnas derivadas porque las secciones 5-7 dependen de ellas)
# =============================================================================
# Duraciones mayores a este umbral se conservan, pero quedan marcadas
UMBRAL_DURACION_EXTREMA = 300   # minutos (5 horas)


def _registrar(registro: list, paso: str, antes: int, df: pd.DataFrame, decision: str) -> None:
    """Agrega una fila al registro de limpieza con las filas antes y después del paso."""
    registro.append({
        "paso": paso,
        "filas_antes": antes,
        "filas_despues": len(df),
        "filas_eliminadas": antes - len(df),
        "decision": decision,
    })


def _separar_moneda_y_monto(serie: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Separa textos como '$ 2250' o 'INR 1000' en moneda y monto numérico (0 pasa a NaN)."""
    texto = serie.str.strip()
    moneda = texto.str.extract(r"^([^\d\s.,-]+)")[0]
    monto = pd.to_numeric(texto.str.replace(r"\D", "", regex=True), errors="coerce")
    # Un monto 0 significa dato desconocido, no una película sin presupuesto
    monto = monto.where(monto > 0)
    return moneda, monto


def _tabla_sesgo_genero(df: pd.DataFrame) -> pd.DataFrame:
    """Compara cuántas películas tienen cada género con cuántas lo reciben como genero_principal."""
    con_genero = df["generos"].explode().value_counts()
    como_principal = df["genero_principal"].value_counts()
    tabla = pd.DataFrame({"peliculas_con_el_genero": con_genero,
                          "como_genero_principal": como_principal}).fillna(0).astype(int)
    tabla["pct_como_principal"] = (tabla["como_genero_principal"]
                                   / tabla["peliculas_con_el_genero"] * 100).round(1)
    tabla = tabla.sort_values("peliculas_con_el_genero", ascending=False)
    tabla.index.name = "genero"
    return tabla


def preparar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpieza y preparación: duplicados, conversión de tipos, montos, valores
    extremos y variables derivadas. Registra las filas antes y después de cada
    paso y lo exporta a resultados/tablas/registro_limpieza.csv.
    """
    print("\n=== 3-4. LIMPIEZA Y PREPARACIÓN ===")
    df = df.copy()
    registro = []
    n_entrada = df.attrs.get("n_registros_entrada", len(df))
    _registrar(registro, "Carga inicial (sin columnas personales)", n_entrada, df,
               "Datos tal como llegan del diagnóstico; solo se quitaron director, writer y actors")

    # Duplicados: el diagnóstico encontró 0, se deja el control por reproducibilidad.
    # Mismo título y año con distinto ID son películas distintas y se conservan.
    antes = len(df)
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset=["imdb_title_id"], keep="first")
    _registrar(registro, "Duplicados exactos e IDs repetidos", antes, df,
               "Se eliminan si existen; las 235 filas con mismo título y año e ID distinto se conservan")

    # `year` llega como texto por una fila con valor 'TV Movie 2019'
    antes = len(df)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    _registrar(registro, "Conversión de year a número", antes, df,
               "Se eliminan las filas cuyo año no es numérico ('TV Movie 2019')")

    # `date_published` trae filas con solo el año; el análisis temporal usa `year`
    antes = len(df)
    df["date_published"] = pd.to_datetime(df["date_published"], format=FORMATO_FECHA, errors="coerce")
    sin_fecha = int(df["date_published"].isna().sum())
    _registrar(registro, "Conversión de date_published a fecha", antes, df,
               f"{sin_fecha} fechas con solo el año quedan vacías (NaT); se conservan y el tiempo se analiza con year")

    # Montos: vienen con prefijo de moneda y no se pueden comparar entre monedas
    for col in COLUMNAS_MONETARIAS:
        antes = len(df)
        ceros = int((pd.to_numeric(df[col].str.replace(r"\D", "", regex=True), errors="coerce") == 0).sum())
        df[f"{col}_moneda"], df[f"{col}_monto"] = _separar_moneda_y_monto(df[col])
        con_dato = int(df[f"{col}_monto"].notna().sum())
        _registrar(registro, f"Separación de moneda y monto en {col}", antes, df,
                   f"{con_dato} filas con monto ({con_dato / len(df):.1%}); {ceros} montos en 0 pasan a "
                   "faltante; las monedas no se mezclan; se conservan las filas sin monto")

    # Duración: se eliminan solo valores imposibles; los extremos se conservan con advertencia
    antes = len(df)
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
    df = df[df["duration"] > 0]
    df["duracion_extrema"] = df["duration"] > UMBRAL_DURACION_EXTREMA
    extremas = int(df["duracion_extrema"].sum())
    _registrar(registro, "Validación de duration", antes, df,
               f"Se eliminan nulos o <= 0; {extremas} películas de más de {UMBRAL_DURACION_EXTREMA} min "
               f"(máx. {df['duration'].max()}) se conservan marcadas en duracion_extrema")

    # Nota y votos son necesarios para responder las 4 preguntas
    antes = len(df)
    df["avg_vote"] = pd.to_numeric(df["avg_vote"], errors="coerce")
    df["votes"] = pd.to_numeric(df["votes"], errors="coerce")
    df = df.dropna(subset=["avg_vote", "votes"])
    _registrar(registro, "Validación de avg_vote y votes", antes, df,
               "Se eliminan películas sin nota o sin votos")

    # Variables derivadas (documentadas en el diccionario de datos)
    antes = len(df)
    df["decada"] = (df["year"] // 10) * 10                      # década de estreno
    df["log_votes"] = np.log10(df["votes"])                      # votos en escala log10
    df["genero_principal"] = df["genre"].str.split(",").str[0].str.strip()
    df["pais_principal"] = df["country"].str.split(",").str[0].str.strip()
    # IMDb lista los géneros en orden alfabético, así que genero_principal es el
    # primero del abecedario y no el más representativo. `generos` guarda la
    # lista completa para analizar cada película en todos sus géneros (df.explode).
    df["generos"] = df["genre"].str.split(",").apply(lambda lista: [g.strip() for g in lista])
    pct_alfabetico = df["generos"].apply(lambda lista: lista == sorted(lista)).mean() * 100
    sin_pais = int(df["pais_principal"].isna().sum())
    _registrar(registro, "Variables derivadas", antes, df,
               f"Se crean decada, log_votes, genero_principal, pais_principal y generos; "
               f"{sin_pais} películas sin país se conservan (quedan fuera solo de las tablas por país)")

    tabla = pd.DataFrame(registro).set_index("paso")
    print(tabla[["filas_antes", "filas_despues", "filas_eliminadas"]].to_string())
    print(f"\nRegistros de entrada: {n_entrada:,} | Registros finales: {len(df):,}")
    print(f"ADVERTENCIA: {pct_alfabetico:.1f}% de las películas lista sus géneros en orden alfabético; "
          "genero_principal favorece a los géneros que empiezan con A, B o C (ver sesgo_genero_principal.csv)")
    guardar_tabla(tabla, "registro_limpieza")
    guardar_tabla(_tabla_sesgo_genero(df), "sesgo_genero_principal")
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
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def generar_graficos(df: pd.DataFrame, resultados: dict) -> None:
    """Genera y exporta los 6 gráficos obligatorios en formato PNG."""
    # Asegurar que la carpeta de figuras exista usando la variable global del script
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)

    # 1. Distribución de una variable numérica (Nota promedio)
    plt.figure(figsize=(8, 5))
    df['avg_vote'].hist(bins=20, color='skyblue', edgecolor='black')
    plt.title('1. Distribución de la Calificación Promedio (avg_vote)')
    plt.xlabel('Calificación Promedio')
    plt.ylabel('Frecuencia (Cantidad de Películas)')
    plt.savefig(DIR_FIGURAS / '1_distribucion_nota.png', bbox_inches='tight')
    plt.close()

    # 2. Valores extremos (Duración)
    plt.figure(figsize=(8, 5))
    plt.boxplot(df['duration'].dropna(), orientation='horizontal', patch_artist=True)
    plt.title('2. Distribución y Valores Extremos de la Duración')
    plt.xlabel('Duración (minutos)')
    plt.yticks([1], ['Películas'])
    plt.savefig(DIR_FIGURAS / '2_valores_extremos_duracion.png', bbox_inches='tight')
    plt.close()

    # 3. Dispersión por grupos (Relación Nota vs Votos por los 5 géneros más comunes)
    # Se filtran solo los top 5 para evitar un gráfico ilegible por exceso de categorías
    top_5_generos = resultados["frecuencia_genero"].head(5).index
    df_top_generos = df[df['genero_principal'].isin(top_5_generos)]
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_top_generos, x='log_votes', y='avg_vote', hue='genero_principal', alpha=0.6)
    plt.title('3. Dispersión de Calificación vs Votos (Top 5 Géneros Principales)')
    plt.xlabel('Votos (Escala Logarítmica - log_votes)')
    plt.ylabel('Calificación Promedio')
    plt.legend(title='Género Principal')
    plt.savefig(DIR_FIGURAS / '3_dispersion_grupos_genero.png', bbox_inches='tight')
    plt.close()

    # 4. Frecuencias o comparación de categorías (Top 10 Países)
    plt.figure(figsize=(10, 6))
    df_pais = resultados["frecuencia_pais"].head(10)
    plt.bar(df_pais.index, df_pais['frecuencia'], color='coral', edgecolor='black')
    plt.title('4. Frecuencia de Películas por País de Origen (Top 10)')
    plt.xlabel('País Principal')
    plt.ylabel('Cantidad de Películas')
    plt.xticks(rotation=45, ha='right')
    plt.savefig(DIR_FIGURAS / '4_frecuencias_paises.png', bbox_inches='tight')
    plt.close()

    # 5. Relación entre variables (Duración vs Calificación)
    plt.figure(figsize=(10, 6))
    plt.scatter(df['duration'], df['avg_vote'], alpha=0.3, color='purple')
    plt.title('5. Relación entre Duración y Calificación Promedio')
    plt.xlabel('Duración (minutos)')
    plt.ylabel('Calificación Promedio')
    plt.savefig(DIR_FIGURAS / '5_relacion_duracion_nota.png', bbox_inches='tight')
    plt.close()

    # 6. Gráfico del equipo: Evolución de la duración a lo largo de los años (Pregunta 3)
    plt.figure(figsize=(10, 6))
    df_evolucion = resultados["evolucion_decada"]
    plt.plot(df_evolucion.index, df_evolucion['duracion_mediana'], marker='o', linestyle='-', color='green', linewidth=2)
    plt.title('6. Evolución de la Duración Mediana de las Películas por Década')
    plt.xlabel('Década')
    plt.ylabel('Duración Mediana (minutos)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(DIR_FIGURAS / '6_pregunta_equipo_evolucion_duracion.png', bbox_inches='tight')
    plt.close()
    
    print("  [figuras] 6 gráficos generados y guardados en resultados/figuras/")


# =============================================================================
# PROGRAMA PRINCIPAL
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="EDA IMDb movies — Laboratorio 4")
    parser.add_argument("--ruta", type=Path, default=RUTA_DATOS_POR_DEFECTO,
                        help="ruta al CSV de entrada (por defecto dataset/IMDb movies.csv)")
    args = parser.parse_args()
    np.random.seed(SEMILLA)

    inicializar_carpetas()

    df_original = cargar_datos(args.ruta)
    df_limpio = preparar_datos(df_original)
    resultados = ejecutar_analisis_estadistico(df_limpio)
    generar_graficos(df_limpio, resultados)

    resumen_ejecucion(df_limpio, df_original.attrs.get("n_registros_entrada", len(df_original))) 

    print("\nProceso finalizado correctamente.")


if __name__ == "__main__":
    main()
