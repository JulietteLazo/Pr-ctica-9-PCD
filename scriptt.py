from __future__ import annotations

import datetime as _dt
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# CONSTANTES
# ---------------------------------------------------------------------------

ARCHIVOS = [
    "emisiones-2016.csv",
    "emisiones-2017.csv",
    "emisiones-2018.csv",
    "emisiones-2019.csv",
]

# Columnas obligatorias básicas
COL_ID = ["ESTACION", "MAGNITUD", "AÑO", "MES"]
# Patrón para columnas de días: D01 … D31 (algunas pueden faltar en febrero)
PATRON_DIAS = r"^D\d{2}$"

# Intentar mapear códigos de MAGNITUD → nombre legible (añádase según necesidad)
MAPA_MAGNITUD = {
    1: "Dióxido de azufre (SO₂)",
    6: "Monóxido de carbono (CO)",
    7: "Monóxido de nitrógeno (NO)",
    8: "Dióxido de nitrógeno (NO₂)",
    9: "Partículas < 10 µm (PM₁₀)",
    10: "Óxidos de nitrógeno (NOₓ)",
    12: "Ozono (O₃)",
    14: "Hidrocarburos totales (T.O.)",
    # … completar según catálogo municipal …
}

# ---------------------------------------------------------------------------
# CARGA Y LIMPIEZA DE DATOS
# ---------------------------------------------------------------------------

def _leer_csv(ruta: Path | str) -> pd.DataFrame:
    """Lee un archivo CSV garantizando los tipos adecuados."""
    df = pd.read_csv(
        ruta,
        sep=";",  # los ficheros municipales suelen venir con ;
        decimal=",",  # y coma decimal en vez de punto
        dtype={
            "ESTACION": "category",
            "MAGNITUD": "int32",
            "AÑO": "int16",
            "MES": "int8",
        },
        na_values=["", "NA", "-", "--", "NoData"],
    )
    return df


def cargar_datos(archivos: list[str] | None = None) -> pd.DataFrame:
    """Carga y concatena los archivos de emisiones, devolviendo un *DataFrame* limpio.

    Pasos aplicados:
    1. Concatena los cuatro años.
    2. Filtra solo las columnas relevantes.
    3. Convierte el formato de ancho (una columna por día) a largo (una fila por
       observación diaria).
    4. Construye la columna ``FECHA`` (*datetime64[ns]*) y descarta fechas
       inválidas.
    5. Ordena por estación, contaminante y fecha.
    """
    if archivos is None:
        archivos = ARCHIVOS

    # ---------- 1) Lectura y concatenación ----------
    df = pd.concat([_leer_csv(RUTA_DATOS / f) for f in archivos], ignore_index=True)

    # ---------- 2) Filtro de columnas ----------
    columnas_dias = [c for c in df.columns if pd.Series(c).str.contains(PATRON_DIAS).any()]
    columnas_utiles = COL_ID + columnas_dias
    df = df[columnas_utiles]

    # ---------- 3) Re‑estructuración (melt) ----------
    df = df.melt(
        id_vars=COL_ID,
        value_vars=columnas_dias,
        var_name="DIA",
        value_name="VALOR",
    )

    # Eliminar posibles filas completamente vacías
    df.dropna(subset=["VALOR"], how="all", inplace=True)

    # DIA: convertir "D01" → 1 … etc.
    df["DIA"] = df["DIA"].str[1:].astype("int8")

    # ---------- 4) Construcción de columna FECHA ----------
    df["FECHA"] = pd.to_datetime(
        dict(year=df["AÑO"], month=df["MES"], day=df["DIA"]), errors="coerce"
    )
    # Eliminar fechas inválidas (NaT)
    df = df[~pd.isna(df["FECHA"])]

    # ---------- 5) Orden ----------
    df.sort_values(["ESTACION", "MAGNITUD", "FECHA"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


# ---------------------------------------------------------------------------
# UTILIDADES DE CONSULTA
# ---------------------------------------------------------------------------

def estaciones_y_contaminantes(df: pd.DataFrame) -> Tuple[pd.Index, pd.Index]:
    """Devuelve un par *(estaciones, contaminantes)* disponibles en *df*."""
    estaciones = df["ESTACION"].unique()
    contaminantes = df["MAGNITUD"].unique()
    return estaciones, contaminantes


def emisiones_por_rango(
    df: pd.DataFrame,
    estacion: str | int,
    magnitud: int,
    fecha_ini: str | _dt.date | pd.Timestamp,
    fecha_fin: str | _dt.date | pd.Timestamp,
) -> pd.Series:
    """Serie de emisiones para ``estacion`` y ``magnitud`` en el rango [fecha_ini, fecha_fin]."""
    rango = (pd.to_datetime(fecha_ini), pd.to_datetime(fecha_fin))
    mask = (
        (df["ESTACION"] == str(estacion))
        & (df["MAGNITUD"] == magnitud)
        & (df["FECHA"] >= rango[0])
        & (df["FECHA"] <= rango[1])
    )
    return df.loc[mask].set_index("FECHA")["VALOR"].astype(float)


# ---------------------------------------------------------------------------
# RESÚMENES DESCRIPTIVOS
# ---------------------------------------------------------------------------

def resumen_general(df: pd.DataFrame) -> pd.DataFrame:
    """Resumen descriptivo (count, mean, std, min, …) para cada contaminante."""
    return df.groupby("MAGNITUD")["VALOR"].describe()


def resumen_por_distrito(df: pd.DataFrame, mapa_estacion_distrito: dict[str, str]) -> pd.DataFrame:
    """Resumen descriptivo por contaminante y distrito.

    Se necesita proporcionar un *dict* que mapée código de estación → nombre de distrito.
    """
    if "DISTRITO" not in df.columns:
        df = df.assign(DISTRITO=df["ESTACION"].map(mapa_estacion_distrito))
    return df.groupby(["DISTRITO", "MAGNITUD"])["VALOR"].describe()


def resumen_estacion_contaminante(
    df: pd.DataFrame,
    estacion: str | int,
    magnitud: int,
) -> pd.Series:
    """Resumen descriptivo para una pareja estación / contaminante."""
    mask = (df["ESTACION"] == str(estacion)) & (df["MAGNITUD"] == magnitud)
    return df.loc[mask, "VALOR"].describe()


# ---------------------------------------------------------------------------
# MEDIAS MENSUALES
# ---------------------------------------------------------------------------

def medias_mensuales_contaminante(
    df: pd.DataFrame, magnitud: int, anio: int
) -> pd.DataFrame:
    """Emisiones medias mensuales del *magnitud* en el *anio* para todas las estaciones."""
    mask = (df["AÑO"] == anio) & (df["MAGNITUD"] == magnitud)
    return (
        df.loc[mask]
        .groupby(["ESTACION", "MES"], observed=True)["VALOR"]
        .mean()
        .unstack(level="MES")
        .sort_index()
    )


def medias_mensuales_estacion(df: pd.DataFrame, estacion: str | int) -> pd.DataFrame:
    """DataFrame (contaminantes × meses) con medias mensuales en la *estacion*."""
    mask = df["ESTACION"] == str(estacion)
    return (
        df.loc[mask]
        .groupby(["MAGNITUD", "MES"], observed=True)["VALOR"]
        .mean()
        .unstack(level="MES")
        .sort_index()
    )


# ---------------------------------------------------------------------------
# DEMOSTRACIÓN RÁPIDA (solo si se ejecuta como script)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = cargar_datos()

    # 1) Mostrar estaciones y contaminantes disponibles
    ests, mags = estaciones_y_contaminantes(df)
    print("Estaciones disponibles ({}): {}".format(len(ests), ", ".join(ests)))
    print("Contaminantes disponibles (códigos): {}".format(", ".join(map(str, mags))))

    # 2) Resumen general
    print("\nResumen descriptivo general por contaminante:\n")
    print(resumen_general(df))

    # 3) Ejemplo de función emisiones_por_rango
    ejemplo = emisiones_por_rango(df, estacion=28079004, magnitud=8, fecha_ini="2019-01-01", fecha_fin="2019-01-31")
    print("\nEjemplo emisiones estación 28079004 – NO₂ – enero 2019:")
    print(ejemplo.head())