import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import soccerdata as sd
import ScraperFC as sfc
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import os

st.set_page_config(page_title="EDA – Football Player Analysis", page_icon="⚽", layout="wide")

st.title("⚽ EDA y análisis de similitud de jugadores")
st.markdown("""
Esta aplicación reproduce el análisis desarrollado en el Jupyter original. La lógica del análisis se mantiene: extracción de datos de FBref, unificación de datasets, selección de variables, tratamiento de valores nulos, estandarización, PCA y comparación de jugadores mediante similitud coseno y distancia euclídea.

La aplicación añade únicamente una capa de visualización y explicación para que el análisis pueda consultarse de forma interactiva.
""")

with st.sidebar:
    st.header("Navegación")
    st.info("Los parámetros del análisis se mantienen tal como están definidos en el Jupyter original.")
    st.markdown("**Fuente:** FBref mediante `soccerdata` y datos de Transfermarkt mediante `ScraperFC`.")
    st.markdown("**Temporada principal:** Premier League 2025/26")

with st.expander("Código original del análisis", expanded=False):
    st.markdown("Las celdas de análisis se han trasladado al script sin importar ni ejecutar el notebook como módulo.")

st.header("1. Obtención de los datos")
st.write("Primero se cargan el calendario, las estadísticas de los equipos y las estadísticas de jugadores de la Premier League.")

# ===== CELDA 0 =====
import os
import pandas as pd
import numpy as np
import seaborn as sns
#import LanusStats  as ls
import soccerdata as sd
import ScraperFC as sfc
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity



fbref = sd.FBref('ENG-Premier League', '2024')

games = fbref.read_schedule()
team_season_stats = fbref.read_team_season_stats(stat_type="standard")
player_season_stats = fbref.read_player_season_stats(stat_type="standard")
print(player_season_stats.head())

st.subheader("Primer vistazo a las estadísticas de jugadores")
st.dataframe(player_season_stats.head(), use_container_width=True)

print(player_season_stats.columns)

player_season_stats.head()

st.header("2. Estadísticas complementarias y unificación")
st.write("Además de las estadísticas estándar, se incorporan shooting, playing time, goalkeeper y miscellaneous para construir una única tabla de análisis.")

player_season_stats_shooting = fbref.read_player_season_stats(stat_type="shooting")
player_season_stats_playing_time = fbref.read_player_season_stats(stat_type="playing_time")
player_season_stats_keeper = fbref.read_player_season_stats(stat_type="keeper")
player_season_stats_misc = fbref.read_player_season_stats(stat_type="misc")

print(player_season_stats.columns, player_season_stats_shooting.columns, player_season_stats_playing_time.columns, player_season_stats_keeper.columns, player_season_stats_misc.columns)




print(player_season_stats.columns, player_season_stats_shooting.columns, player_season_stats_playing_time.columns, player_season_stats_keeper.columns, player_season_stats_misc.columns)


print(f"Indices: {player_season_stats.index}")
print(f"Columnas {player_season_stats.columns}")

claves = ["player", "nation", "born", "pos", "age"]

dfs = [
    player_season_stats,
    player_season_stats_shooting,
    player_season_stats_playing_time,
    player_season_stats_keeper,
    player_season_stats_misc
]

for df_temp in dfs:
    if "player" not in df_temp.columns:
        df_temp.reset_index(inplace=True)

for df_temp in dfs:
    if isinstance(df_temp.columns, pd.MultiIndex):
        df_temp.columns = [
            f"{col[0]}_{col[1]}" if col[1] != "" else col[0]
            for col in df_temp.columns
        ]

columnas_repetidas = ["team", "season", "league"]

for df_temp in dfs[1:]:
    df_temp.drop(
        columns=[col for col in columnas_repetidas if col in df_temp.columns],
        inplace=True
    )
    
df = player_season_stats.merge(
    player_season_stats_shooting,
    on=claves,
    how="left"
)

df = df.merge(
    player_season_stats_playing_time,
    on=claves,
    how="left"
)

df = df.merge(
    player_season_stats_keeper,
    on=claves,
    how="left"
)

df = df.merge(
    player_season_stats_misc,
    on=claves,
    how="left"
)

print(df.info())
print(df[(df["team"]=="Manchester City")])

st.success(f"Dataset unificado: {df.shape[0]} filas y {df.shape[1]} columnas.")
st.dataframe(df.head(), use_container_width=True)

fbref2 = sd.FBref('FRA-Ligue 1', '2024')

games2 = fbref2.read_schedule()
team_season_stats2 = fbref2.read_team_season_stats(stat_type="standard")
player_season_stats2 = fbref2.read_player_season_stats(stat_type="standard")
print(player_season_stats2.head())

st.subheader("Primer vistazo a las estadísticas de jugadores - Serie A")
st.dataframe(player_season_stats2.head(), use_container_width=True)

print(player_season_stats2.columns)

player_season_stats2.head()

st.header("2. Estadísticas complementarias y unificación")
st.write("Además de las estadísticas estándar, se incorporan shooting, playing time, goalkeeper y miscellaneous para construir una única tabla de análisis.")

player_season_stats_shooting2 = fbref2.read_player_season_stats(stat_type="shooting")
player_season_stats_playing_time2 = fbref2.read_player_season_stats(stat_type="playing_time")
player_season_stats_keeper2 = fbref2.read_player_season_stats(stat_type="keeper")
player_season_stats_misc2 = fbref2.read_player_season_stats(stat_type="misc")


dfs2 = [
    player_season_stats2,
    player_season_stats_shooting2,
    player_season_stats_playing_time2,
    player_season_stats_keeper2,
    player_season_stats_misc2
]


for df_temp in dfs2:
    if "player" not in df_temp.columns:
        df_temp.reset_index(inplace=True)

for df_temp in dfs2:
    if isinstance(df_temp.columns, pd.MultiIndex):
        df_temp.columns = [
            f"{col[0]}_{col[1]}" if col[1] != "" else col[0]
            for col in df_temp.columns
        ]

columnas_repetidas = ["team", "season", "league"]

for df_temp in dfs2[1:]:
    df_temp.drop(
        columns=[col for col in columnas_repetidas if col in df_temp.columns],
        inplace=True
    )


df2 = player_season_stats2.merge(
    player_season_stats_shooting2,
    on=claves,
    how="left"
)

df2 = df2.merge(
    player_season_stats_playing_time2,
    on=claves,
    how="left"
)

df2 = df2.merge(
    player_season_stats_keeper2,
    on=claves,
    how="left"
)

df2 = df2.merge(
    player_season_stats_misc2,
    on=claves,
    how="left"
)

print(df2.info())

st.success(f"Dataset Serie A unificado: {df2.shape[0]} filas y {df2.shape[1]} columnas.")
st.dataframe(df2.head(), use_container_width=True)


df = df.drop(columns=["league", "nation", "age", "born"])

df = pd.concat(
    [df, df2],
    ignore_index=True
)
df

st.success(f"Dataset final: {df.shape[0]} filas y {df.shape[1]} columnas.")



st.header("3. Reducción de dimensionalidad")
st.markdown("Momento de intentar reducir la dimensionalidad, dando prioridad a las variables más relevantes.")
st.markdown("**Variables consideradas más relevantes, eliminando redundancias a la vez, acabamos con 22 variables.**")

variables_pca = [
    "Per 90 Minutes_Gls",
    "Per 90 Minutes_Ast",
    "Per 90 Minutes_G+A",
    "Per 90 Minutes_G-PK",
    "Per 90 Minutes_G+A-PK",
    
    "Standard_Sh/90",
    "Standard_SoT/90",
    "Standard_SoT%",
    "Standard_G/Sh",
    "Standard_G/SoT",
    
    "Playing Time_Min%",
    
    "Team Success_PPM",
    "Team Success_+/-90",
    "Team Success_On-Off",
    
    "Performance_Fls",
    "Performance_Fld",
    "Performance_Off",
    "Performance_Crs",
    "Performance_Int",
    "Performance_TklW",
    
    "Performance_CrdY_y",
    "Performance_CrdR_y"
]

df_sin_porteros = df[df["pos"] != "GK"].copy()


FILTRAR_POR_POSICION = False  
POSICION_FILTRO = "FW"        

if FILTRAR_POR_POSICION:
    df_sin_porteros = df_sin_porteros[
        df_sin_porteros["pos"].str.contains(POSICION_FILTRO, na=False)
    ].copy()

df_pca_data = df_sin_porteros[variables_pca].copy()

st.write(f"Jugadores tras excluir porteros y aplicar el filtro definido: {df_pca_data.shape[0]}")

df_num = df_pca_data.select_dtypes(include=['number'])
df_num.index = df_sin_porteros['player']


UMBRAL_NAN = 10

na_counts = df_num.isna().sum()
cols_pocos_na = na_counts[(na_counts > 0) & (na_counts < UMBRAL_NAN)].index.tolist()
cols_muchos_na = na_counts[na_counts >= UMBRAL_NAN].index.tolist()

print(f"Columnas con <{UMBRAL_NAN} NaN (se eliminan jugadores): {cols_pocos_na}")
print(f"Columnas con >={UMBRAL_NAN} NaN (se imputa la moda): {cols_muchos_na}")

df_num = df_num.dropna(subset=cols_pocos_na)


for col in cols_muchos_na:
    moda = df_num[col].mode(dropna=True)[0]
    df_num[col] = df_num[col].fillna(moda)

print(f"Jugadores restantes tras el tratamiento de nulos: {df_num.shape[0]}")
df_num = df_num[~df_num.index.duplicated(keep="first")]

st.write(f"Matriz numérica para PCA: {df_num.shape[0]} jugadores × {df_num.shape[1]} variables.")

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_num)

pca = PCA(n_components=10)
df_pca= pca.fit_transform(df_scaled)

print(pca.n_components_)

st.metric("Componentes PCA", pca.n_components_)

pesos = pd.DataFrame(
    pca.components_.T, 
    columns=[f"PC{i+1}" for i in range(pca.components_.shape[0])],
    index= df_num.columns
)
print(pesos)

st.subheader("Pesos (loadings) de las variables")
st.dataframe(pesos, use_container_width=True)


varianza = pca.explained_variance_ratio_
acumulada = np.cumsum(varianza)

plt.figure(figsize=(10, 6))

plt.bar(
    range(1, len(varianza) + 1),
    varianza,
    label="Varianza individual"
)

plt.plot(
    range(1, len(acumulada) + 1),
    acumulada,
    marker="o",
    label="Varianza acumulada"
)

plt.axhline(
    0.90,
    linestyle="--",
    label="90% de varianza"
)

plt.xlabel("Componentes principales")
plt.ylabel("Varianza explicada")
plt.title("Scree Plot del PCA")
plt.xticks(range(1, len(varianza) + 1))
plt.legend()


st.pyplot(plt.gcf(), clear_figure=True)

plt.figure(figsize=(14, 9))

plt.imshow(
    pesos,
    aspect="auto"
)

plt.colorbar(label="Loading")

plt.xticks(
    range(len(pesos.columns)),
    pesos.columns
)

plt.yticks(
    range(len(pesos.index)),
    pesos.index
)

plt.xlabel("Componentes principales")
plt.ylabel("Variables")
plt.title("Importancia de las variables en cada componente")

plt.tight_layout()

st.pyplot(plt.gcf(), clear_figure=True)

st.header("4. Unificación con ELO y valor contractual")
st.markdown("Esta sección conserva el flujo original para incorporar la valoración ELO y la información contractual de Transfermarkt.")

#clubelo = sd.ClubElo()


st.markdown("Definimos 1 de junio de 2026 como el final de la temporada 25/26.")

#current_elo = clubelo.read_by_date('2026-06-01')

#tm = sfc.Transfermarkt()

#import ScraperFC.transfermarkt as transfermarkt
import builtins
import re

float_original = builtins.float


def float_patched(value):
    if isinstance(value, str) and "ft" in value and "in" in value:
        match = re.search(
            r"(\d+)\s*ft\s*(\d+)?\s*in",
            value
        )

        if match:
            pies = float_original(match.group(1))
            pulgadas = float_original(match.group(2) or 0)

            return pies * 0.3048 + pulgadas * 0.0254

    return float_original(value)


builtins.float = float_patched

# valor_jugadores = tm.scrape_players(
#     "25/26",
#     "England Premier League"
# )

# print(valor_jugadores.head())

# st.dataframe(valor_jugadores.head(), use_container_width=True)
builtins.float = float_original


st.header("5. Similitud entre jugadores mediante PCA")
st.write("A partir de los componentes PCA se reconstruye la matriz de puntuaciones por jugador. Después se calcula la similitud coseno tomando como referencia a Kevin De Bruyne, exactamente como en el Jupyter.")

df_pca = pca.fit_transform(df_scaled)
df_pca = pd.DataFrame(
    df_pca, 
    index =  df_num.index,
    columns=[f"PC{i+1}" for i in range(df_pca.shape[1])]
)

#df_sin_porteros = df_sin_porteros.reset_index(drop=True)
#df_pca = df_pca.reset_index(drop=True)


saka = df_pca.loc["Kevin De Bruyne"].values.reshape(1,-1)

saka_cosine=cosine_similarity(saka,df_pca)

print(saka_cosine.shape)

print("Shape de Saka:", saka.shape)
print("Shape del PCA:", df_pca.shape)

similitud = pd.Series(
    saka_cosine[0],
    index=df_num.index
)

similitud = similitud.sort_values(ascending=False)

similitud = similitud[similitud.index != "Kevin De Bruyne"]

top_5 = similitud.head(10)

print(top_5)

st.subheader("Jugadores más similares según similitud coseno")

tabla_similares = pd.DataFrame({
    "Jugador": top_5.index,
    "Similitud": top_5.values
})

tabla_similares
st.dataframe(tabla_similares, use_container_width=True)

jugadores = ["Kevin De Bruyne"] + top_5.index.tolist()
componentes = df_pca.columns.tolist()


datos = df_pca.loc[jugadores, componentes]
print(datos)

st.subheader("Perfil PCA de los jugadores comparados")

fig, ax = plt.subplots(figsize=(12, 8))

y = np.arange(len(componentes))

for jugador in jugadores:
    ax.scatter(
        datos.loc[jugador, componentes],
        y,
        s=80,
        label=jugador
    )

ax.axvline(0, linewidth=1, linestyle="--")


ax.set_yticks(y)
ax.set_yticklabels(componentes)

ax.set_xlabel("Valor del componente PCA")
ax.set_ylabel("Componentes")
ax.set_title(
    "Perfil PCA: Bukayo Saka vs jugadores similares",
    fontsize=16
)

ax.legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

plt.tight_layout()

st.pyplot(plt.gcf(), clear_figure=True)

limite = np.abs(datos.values).max()

fig, ax = plt.subplots(figsize=(14, 6))

im = ax.imshow(
    datos.values,
    aspect="auto",
    cmap="RdBu_r",
    vmin=-limite,
    vmax=limite
)

ax.set_yticks(range(len(jugadores)))
ax.set_yticklabels(jugadores)

ax.set_xticks(range(len(componentes)))
ax.set_xticklabels(componentes)

for i in range(len(jugadores)):
    for j in range(len(componentes)):
        valor = datos.iloc[i, j]

        ax.text(
            j,
            i,
            f"{valor:.2f}",
            ha="center",
            va="center"
        )

ax.set_xlabel("Componentes PCA")
ax.set_ylabel("Jugador")

ax.set_title(
    "Perfil PCA: Bukayo Saka vs jugadores similares",
    fontsize=16
)

plt.colorbar(im, ax=ax, label="Valor PCA")

plt.tight_layout()

st.pyplot(plt.gcf(), clear_figure=True)

st.markdown("Como en el análisis original, se utiliza MinMaxScaler únicamente para transformar las componentes a un rango positivo para poder representarlas en el radar.")

scaler_radar = MinMaxScaler()

df_pca_minmax = pd.DataFrame(
    scaler_radar.fit_transform(datos),
    index = datos.index,
    columns=datos.columns
)



st.subheader("Radar de perfiles")

N = len(componentes)

angulos = np.linspace(
    0,
    2 * np.pi,
    N,
    endpoint=False
).tolist()

angulos += angulos[:1]

fig, ax = plt.subplots(
    figsize=(10, 10),
    subplot_kw=dict(polar=True)
)

for jugador in jugadores:

    valores = df_pca_minmax.loc[
        jugador,
        componentes
    ].tolist()

    valores += valores[:1]

    ax.plot(
        angulos,
        valores,
        linewidth=2,
        label=jugador
    )

    ax.fill(
        angulos,
        valores,
        alpha=0.08
    )

ax.set_xticks(angulos[:-1])
#ax.set_xticklabels(etiquetas)

ax.set_ylim(0, 1)

ax.set_title(
    "Perfil PCA: Bukayo Saka vs jugadores similares",
    fontsize=16,
    pad=25
)

ax.legend(
    loc="upper right",
    bbox_to_anchor=(1.35, 1.10)
)


st.pyplot(plt.gcf(), clear_figure=True)

st.header("6. Comparación: similitud coseno vs distancia euclídea")
st.markdown("El MinMaxScaler de la sección anterior es solo para la representación del radar; no interviene en las medidas de similitud. Aquí se comparan los rankings sobre las mismas componentes PCA sin ese reescalado.")

from sklearn.metrics.pairwise import euclidean_distances

saka_euclidean = euclidean_distances(saka, df_pca)

print(saka_euclidean.shape)

distancia = pd.Series(
    saka_euclidean[0],
    index=df_num.index
)

distancia = distancia.sort_values(ascending=True)  # menor distancia = más parecido
distancia = distancia[distancia.index != "Kevin De Bruyne"]

N_SIMILARES = len(top_5)  

top_euclidean = distancia.head(N_SIMILARES)
print(top_euclidean)

tabla_comparacion = pd.DataFrame({
    "Jugador (Coseno)": top_5.index,
    "Similitud (Coseno)": top_5.values,
    "Jugador (Euclídea)": top_euclidean.index,
    "Distancia (Euclídea)": top_euclidean.values,
})

tabla_comparacion

st.subheader("Comparación de rankings")
st.dataframe(tabla_comparacion, use_container_width=True)

coincidencias = set(top_5.index) & set(top_euclidean.index)
print(f"Jugadores que coinciden en ambos métodos: {coincidencias if coincidencias else 'ninguno'}")

st.success(f"Coincidencias entre ambos métodos: {coincidencias if coincidencias else 'ninguna'}")

st.caption("Aplicación Streamlit generada a partir de EDA_01.ipynb. La lógica analítica original se conserva; Streamlit se utiliza como capa de presentación.")
