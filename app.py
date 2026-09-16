import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import soccerdata as sd
import ScraperFC as sfc
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import os
import plotly.express as px

@st.cache_data(show_spinner="Cargando y procesando datos de FBref...")
def cargar_y_procesar_datos():
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


    #ligas = ["ENG-Premier League", ]
    fbref = sd.FBref('Big 5 European Leagues Combined', '2026')

    games = fbref.read_schedule()
    team_season_stats = fbref.read_team_season_stats(stat_type="standard")
    player_season_stats = fbref.read_player_season_stats(stat_type="standard")
    print(player_season_stats.head())

    print(player_season_stats.columns)

    player_season_stats.head()

    player_season_stats_shooting = fbref.read_player_season_stats(stat_type="shooting")
    player_season_stats_playing_time = fbref.read_player_season_stats(stat_type="playing_time")
    player_season_stats_keeper = fbref.read_player_season_stats(stat_type="keeper")
    player_season_stats_misc = fbref.read_player_season_stats(stat_type="misc")

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

 
    df = df.drop(columns=["league", "born"])


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


    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_num)

    pca = PCA(n_components=10)
    df_pca= pca.fit_transform(df_scaled)

    print(pca.n_components_)


    pesos = pd.DataFrame(
        pca.components_.T, 
        columns=[f"PC{i+1}" for i in range(pca.components_.shape[0])],
        index= df_num.columns
    )
    print(pesos)


    varianza = pca.explained_variance_ratio_
    acumulada = np.cumsum(varianza)


    #clubelo = sd.ClubElo()



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

    builtins.float = float_original


    df_pca = pca.fit_transform(df_scaled)
    df_pca = pd.DataFrame(
        df_pca, 
        index =  df_num.index,
        columns=[f"PC{i+1}" for i in range(df_pca.shape[1])]
    )

    componentes = df_pca.columns.tolist()

    K_RANGO = range(2, 11)
    inercias = []

    for k in K_RANGO:
        kmeans_tmp = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans_tmp.fit(df_pca[componentes])
        inercias.append(kmeans_tmp.inertia_)

    diferencias = np.diff(inercias)
    diferencias2 = np.diff(diferencias)
    N_CLUSTERS = list(K_RANGO)[int(np.argmax(diferencias2)) + 1] if len(diferencias2) > 0 else 3

    print(f"Inercias por k: {inercias}")
    print(f"Número óptimo de clusters (codo): {N_CLUSTERS}")

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    clusters_kmeans = pd.Series(
        kmeans.fit_predict(df_pca[componentes]),
        index=df_pca.index,
        name="Cluster",
    )

    print(clusters_kmeans.value_counts())

    return df_sin_porteros, df_num, df_pca, componentes, K_RANGO, inercias, N_CLUSTERS, clusters_kmeans


df_sin_porteros, df_num, df_pca, componentes, K_RANGO, inercias, N_CLUSTERS, clusters_kmeans = cargar_y_procesar_datos()



st.set_page_config(
    page_title="Radar de Jugadores — Similitud & Comparador",
    layout="wide",
)

# ---- Estilos ----
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

    :root {
        --bg: #0a100d;
        --panel: #121a16;
        --panel-alt: #16201b;
        --border: #263029;
        --text: #edf1ee;
        --text-muted: #94a69b;
        --accent: #d9a441;
        --accent-b: #6f93b3;
    }

    .stApp {
        background-color: var(--bg);
        color: var(--text);
    }
    .stApp, p, span, label, div {
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3, h4 {
        font-family: 'Oswald', sans-serif;
        font-weight: 600;
        letter-spacing: 0.01em;
        color: var(--text);
    }

    /* Cabecera */
    .hero {
        padding: 2.2rem 2.4rem;
        border-radius: 4px;
        background: linear-gradient(180deg, var(--panel-alt) 0%, var(--bg) 100%);
        border: 1px solid var(--border);
        border-left: 3px solid var(--accent);
        margin-bottom: 1.6rem;
    }
    .hero-kicker {
        color: var(--accent);
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .hero h1 {
        font-size: 2.5rem;
        margin: 0 0 0.5rem 0;
        line-height: 1.05;
    }
    .hero p {
        font-size: 1.02rem;
        color: var(--text-muted);
        max-width: 68ch;
        line-height: 1.6;
        margin: 0;
    }

    /* Subtítulos de sección */
    h3 {
        border-left: 3px solid var(--accent);
        padding-left: 0.7rem;
        margin-top: 2rem !important;
        font-size: 1.25rem !important;
    }

    /* Fichas de jugador */
    .ficha-jugador {
        padding: 1.1rem 1.3rem;
        border-radius: 4px;
        background-color: var(--panel);
        border: 1px solid var(--border);
        text-align: left;
    }
    .ficha-jugador.ficha-a { border-top: 3px solid var(--accent); }
    .ficha-jugador.ficha-b { border-top: 3px solid var(--accent-b); }
    .ficha-jugador h4 {
        margin: 0 0 0.3rem 0;
        font-size: 1.2rem;
    }
    .ficha-jugador p {
        color: var(--text-muted);
        margin: 0;
        font-size: 0.95rem;
    }

    /* Tablas y métricas */
    div[data-testid="stDataFrame"], div[data-testid="stMetric"] {
        border: 1px solid var(--border);
        border-radius: 4px;
    }
    div[data-testid="stMetric"] {
        background-color: var(--panel);
        padding: 0.6rem 0.8rem;
    }

    /* Pestañas */
    button[data-baseweb="tab"] {
        font-family: 'Oswald', sans-serif;
        font-size: 1.02rem;
    }
    div[data-baseweb="tab-highlight"] {
        background-color: var(--accent) !important;
    }

    /* Cromado por defecto de Streamlit */
    #MainMenu, footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
    <div class="hero-kicker">Similitud de perfiles estadísticos · Big 5 European Leagues</div>
    <h1>Radar de Jugadores</h1>
    <p>
    Cuando el Manchester City perdió a Kevin De Bruyne, su equipo de datos redujo
    decenas de variables de rendimiento a un puñado de componentes y rastreó el
    fútbol europeo buscando los perfiles estadísticamente más parecidos. Rayan Cherki
    fue uno de los nombres que salió de ese análisis, y acabó siendo el fichaje.
    Esta aplicación reproduce esa misma lógica — reducción de dimensionalidad,
    clustering y similitud — sobre datos públicos de las cinco grandes ligas, para
    responder una pregunta muy simple: dado un jugador, ¿quién juega parecido a él?
    </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("Sobre este proyecto y su metodología"):
    st.markdown(
        f"""
        Cada jugador se describe con más de veinte métricas por 90 minutos: producción
        ofensiva, participación defensiva, volumen y eficiencia de tiro, faltas, centros...
        Como muchas de esas métricas están correlacionadas entre sí, se estandarizan y se
        comprimen con un **Análisis de Componentes Principales (PCA)** en
        **{len(componentes)} componentes**, cada uno resumiendo un patrón de juego
        reconocible (producción ofensiva, participación defensiva, eficiencia de
        finalización, creación de juego...).

        Sobre esos componentes se entrena un modelo de **KMeans**, eligiendo el número
        de clusters con el método del codo (**{N_CLUSTERS} clusters** para esta
        temporada), y se calcula la **similitud coseno** entre jugadores para encontrar,
        dado un nombre de referencia, quién tiene el perfil estadístico más parecido.

        El proyecto está inspirado en el caso público del Manchester City y Kevin De
        Bruyne, pero no usa ni reproduce ningún dato interno del club: toda la
        información procede de fuentes públicas (FBref). La similitud es puramente
        estadística — no sustituye el ojo de un scout ni incorpora todavía variables de
        mercado (valor, contrato, cláusula), que quedan como siguiente paso natural del
        proyecto.
        """
    )

COLOR_PANEL = "#121a16"
COLOR_BORDE = "#263029"
COLOR_TEXTO = "#edf1ee"
COLOR_TEXTO_SUAVE = "#94a69b"
COLOR_ACENTO = "#d9a441"
COLOR_ACENTO_B = "#6f93b3"

CMAP_ACENTO = mcolors.LinearSegmentedColormap.from_list(
    "acento_dorado", ["#3a2f1c", COLOR_ACENTO]
)


def estilizar_ejes(fig, ax):
    fig.patch.set_facecolor(COLOR_PANEL)
    ax.set_facecolor(COLOR_PANEL)
    ax.tick_params(colors=COLOR_TEXTO_SUAVE, labelsize=9)
    ax.xaxis.label.set_color(COLOR_TEXTO_SUAVE)
    ax.yaxis.label.set_color(COLOR_TEXTO_SUAVE)
    ax.title.set_color(COLOR_TEXTO)
    for spine in ax.spines.values():
        spine.set_color(COLOR_BORDE)
    ax.grid(axis="x", color=COLOR_BORDE, linewidth=0.6, alpha=0.7)
    return fig, ax


lista_jugadores = sorted(df_num.index.tolist())


def ficha_estatica(nombre_jugador):

    filas = df_sin_porteros[df_sin_porteros["player"] == nombre_jugador]
    if filas.empty:
        fila = None
    else:
        fila = filas.iloc[0]

    def valor(col):
        if fila is None:
            return 0
        return fila[col] if col in fila.index and pd.notna(fila[col]) else 0

    return {
        "Jugador": nombre_jugador,
        "Posición": valor("pos"),
        "Equipo": valor("team"),
        "Edad": valor("age"),
        "Nación": valor("nation"),
        "Cluster": int(clusters_kmeans.loc[nombre_jugador]) if nombre_jugador in clusters_kmeans.index else "-",
        "Goles/90": valor("Per 90 Minutes_Gls"),
        "Asistencias/90": valor("Per 90 Minutes_Ast"),
        "Valor de mercado": 0,
        "Contrato válido hasta": 0,
        "Cláusula de rescisión": 0,
    }


tab_buscador, tab_comparador = st.tabs(["Buscador por similitud", "Comparador de jugadores"])

with tab_buscador:
    st.subheader("1. Elige un jugador de referencia")

    indice_defecto = (
        lista_jugadores.index("Lamine Yamal")
        if "Lamine Yamal" in lista_jugadores
        else 0
    )

    jugador_seleccionado = st.selectbox(
        "Jugador",
        lista_jugadores,
        index=indice_defecto,
        key="selector_similitud",
    )

    n_similares = st.slider("Número de jugadores similares a mostrar", 5, 20, 10)

    if jugador_seleccionado:
        vector_jugador = df_pca.loc[jugador_seleccionado].values.reshape(1, -1)
        similitud_coseno = cosine_similarity(vector_jugador, df_pca)

        similitud = pd.Series(similitud_coseno[0], index=df_num.index)
        similitud = similitud.sort_values(ascending=False)
        similitud = similitud[similitud.index != jugador_seleccionado]

        top_similares = similitud.head(n_similares)

        st.subheader(f"2. Jugadores más similares a {jugador_seleccionado}")

        fig_barras, ax_barras = plt.subplots(figsize=(9, 0.45 * len(top_similares) + 1))
        orden = top_similares.sort_values(ascending=True)
        colores = CMAP_ACENTO(np.linspace(0.35, 0.95, len(orden)))
        ax_barras.barh(orden.index, orden.values, color=colores)
        ax_barras.set_xlabel("Similitud coseno")
        ax_barras.set_xlim(0, 1)
        ax_barras.set_title(f"Jugadores más parecidos a {jugador_seleccionado}")
        for i, v in enumerate(orden.values):
            ax_barras.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9, color=COLOR_TEXTO)
        estilizar_ejes(fig_barras, ax_barras)
        fig_barras.tight_layout()
        st.pyplot(fig_barras, clear_figure=True)

        st.subheader("3. Ficha comparativa de los jugadores similares")

        filas_tabla = []
        for jugador in top_similares.index:
            fila = ficha_estatica(jugador)
            fila["Similitud"] = round(float(top_similares[jugador]), 4)
            filas_tabla.append(fila)

        tabla_similares = pd.DataFrame(filas_tabla)
        columnas_orden = [
            "Jugador", "Similitud", "Posición", "Equipo", "Edad", "Nación", "Cluster",
            "Goles/90", "Asistencias/90", "Valor de mercado",
            "Contrato válido hasta", "Cláusula de rescisión",
        ]
        tabla_similares = tabla_similares[columnas_orden]

        st.dataframe(
            tabla_similares,
            use_container_width=True,
            hide_index=True,
        )


with tab_comparador:
    st.subheader("1. Elige dos jugadores a comparar")

    col_a, col_b = st.columns(2)
    with col_a:
        jugador_a = st.selectbox(
            "Jugador A",
            lista_jugadores,
            index=0,
            key="selector_comparador_a",
        )
    with col_b:
        indice_b = 1 if len(lista_jugadores) > 1 else 0
        jugador_b = st.selectbox(
            "Jugador B",
            lista_jugadores,
            index=indice_b,
            key="selector_comparador_b",
        )

    if jugador_a == jugador_b:
        st.info("Selecciona dos jugadores distintos para comparar.")
    else:
        ficha_a = ficha_estatica(jugador_a)
        ficha_b = ficha_estatica(jugador_b)

        st.subheader("2. Ficha de cada jugador")
        col_ficha_a, col_ficha_b = st.columns(2)
        for col, ficha, clase in ((col_ficha_a, ficha_a, "ficha-a"), (col_ficha_b, ficha_b, "ficha-b")):
            with col:
                st.markdown(
                    f"""
                    <div class="ficha-jugador {clase}">
                    <h4>{ficha['Jugador']}</h4>
                    <p>{ficha['Posición']} · {ficha['Equipo']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.subheader("3. Comparativa de atributos")
        atributos = ["Edad", "Nación", "Equipo", "Posición", "Cluster", "Valor de mercado", "Contrato válido hasta", "Cláusula de rescisión"]
        tabla_atributos = pd.DataFrame(
            {
                "Atributo": atributos,
                jugador_a: [ficha_a[a] for a in atributos],
                jugador_b: [ficha_b[a] for a in atributos],
            }
        )
        st.dataframe(tabla_atributos, use_container_width=True, hide_index=True)

        st.subheader("4. Comparativa de componentes principales (PCA)")

        pcs_a = df_pca.loc[jugador_a, componentes]
        pcs_b = df_pca.loc[jugador_b, componentes]

       # col_hist, col_scatter = st.columns(2)

        #with col_hist:
        st.markdown("**Histograma comparador de PCs**")
        fig_hist, ax_hist = plt.subplots(figsize=(7, 6))
        posiciones = np.arange(len(componentes))
        ancho = 0.38
        ax_hist.barh(posiciones - ancho / 2, pcs_a.values, ancho, label=jugador_a, color=COLOR_ACENTO)
        ax_hist.barh(posiciones + ancho / 2, pcs_b.values, ancho, label=jugador_b, color=COLOR_ACENTO_B)
        ax_hist.axvline(0, color=COLOR_BORDE, linewidth=1, linestyle="--")
        ax_hist.set_yticks(posiciones)
        ax_hist.set_yticklabels(["Producción ofensiva", "Participación defensiva", "Impacto en el rendimiento colectivo", "Volumen de disparo", "Eficiencia de Finaliación", "Creación de Juego / Asistencias", "Disciplina / Faltas", "Creación ofensiva", "Centros", "Impacto colectivo"])
        ax_hist.set_xlabel("Valor del componente PCA")
        leyenda = ax_hist.legend(facecolor=COLOR_PANEL, edgecolor=COLOR_BORDE)
        plt.setp(leyenda.get_texts(), color=COLOR_TEXTO)
        estilizar_ejes(fig_hist, ax_hist)
        fig_hist.tight_layout()
        st.pyplot(fig_hist, clear_figure=True)

        #with col_scatter:
        st.markdown("**Scatterplot 3D (PC1, PC2 vs PC3) de los jugadores seleccionados**")

        df_scatter = df_pca.loc[[jugador_a, jugador_b], ["PC1", "PC2", "PC3"]].copy()
        df_scatter["Jugador"] = df_scatter.index

        fig_scatter = px.scatter_3d(
            df_scatter,
            x="PC1",
            y="PC2",
            z="PC3",
            color="Jugador",
            color_discrete_map={jugador_a: COLOR_ACENTO, jugador_b: COLOR_ACENTO_B},
            hover_name=df_scatter.index,
        )
        fig_scatter.update_layout(
            template="plotly_dark",
            paper_bgcolor=COLOR_PANEL,
            plot_bgcolor=COLOR_PANEL,
            scene=dict(
                xaxis=dict(backgroundcolor=COLOR_PANEL, gridcolor=COLOR_BORDE),
                yaxis=dict(backgroundcolor=COLOR_PANEL, gridcolor=COLOR_BORDE),
                zaxis=dict(backgroundcolor=COLOR_PANEL, gridcolor=COLOR_BORDE),
            ),
            legend=dict(font=dict(color=COLOR_TEXTO)),
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)