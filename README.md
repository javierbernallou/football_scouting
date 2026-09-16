# Radar de Jugadores

Buscador y comparador de jugadores de fútbol por similitud de perfil estadístico, construido sobre datos públicos de las cinco grandes ligas europeas (FBref).

## La idea

Cuando el Manchester City perdió a Kevin De Bruyne, su equipo de datos redujo decenas de variables de rendimiento a un puñado de componentes y rastreó el fútbol europeo buscando los perfiles estadísticamente más parecidos. Rayan Cherki fue uno de los nombres que salió de ese análisis, y acabó siendo el fichaje.

Este proyecto reproduce esa misma lógica —reducción de dimensionalidad, clustering y similitud— sobre datos públicos, para responder una pregunta simple: dado un jugador, ¿quién juega parecido a él?

No usa ni reproduce ningún dato interno de ningún club: toda la información procede de fuentes públicas.

## Qué hace la app

La aplicación tiene dos pestañas:

- **Buscador por similitud**: eliges un jugador de referencia y la app devuelve, ordenados por similitud coseno, los jugadores con el perfil estadístico más parecido dentro de las cinco grandes ligas.
- **Comparador de jugadores**: eliges dos jugadores y la app muestra su ficha (edad, nación, equipo, posición, clúster, rendimiento) lado a lado, junto con un desglose de en qué componentes principales se parecen o se diferencian, y un scatterplot 3D de sus componentes.

## Metodología

1. **Datos**: estadísticas por 90 minutos de jugadores de campo (producción ofensiva, participación defensiva, volumen y eficiencia de tiro, faltas, centros...) obtenidas de FBref a través de la librería `soccerdata`.
2. **Limpieza**: las columnas con pocos valores nulos eliminan al jugador afectado; las columnas con muchos nulos se imputan con la moda.
3. **Estandarización y PCA**: las variables se estandarizan (`StandardScaler`) y se comprimen con un Análisis de Componentes Principales, cada componente resumiendo un patrón de juego reconocible (producción ofensiva, participación defensiva, eficiencia de finalización, creación de juego...).
4. **Clustering**: sobre esos componentes se entrena un `KMeans`, eligiendo el número de clusters con el método del codo.
5. **Similitud**: dado un jugador de referencia, se calcula la similitud coseno frente al resto de la base de datos para encontrar los perfiles más parecidos.

## Instalación

```bash
git clone <url-del-repo>
cd <carpeta-del-repo>
python -m venv .venv
source .venv/bin/activate   # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Uso

```bash
streamlit run app.py
```

La primera carga tarda más porque descarga y procesa los datos de FBref; gracias al cacheo de Streamlit (`@st.cache_data`), las siguientes interacciones son instantáneas mientras no cambien los datos de origen.

## Limitaciones y próximos pasos

- La similitud es puramente estadística: no sustituye el ojo de un scout ni tiene en cuenta el contexto táctico, la edad de desarrollo o el potencial.
- Los campos **Valor de mercado**, **Contrato válido hasta** y **Cláusula de rescisión** todavía no están rellenos: no forman parte de los datos de FBref y son el siguiente paso natural del proyecto (vía Transfermarkt u otra fuente equivalente).
- No hay filtro por posición expuesto en la interfaz todavía, aunque el modelo lo soporta internamente.
- No se muestra el gráfico de varianza explicada del PCA ni un radar comparativo sobre las métricas originales (más allá de los componentes).

## Fuente de datos

- [FBref](https://fbref.com) a través de la librería [`soccerdata`](https://github.com/probberechts/soccerdata).

## Aviso

Proyecto personal con fines educativos y de portfolio, inspirado en un caso público de recomendación de fichajes basado en datos. No está afiliado a ningún club ni utiliza información privada de ninguna organización.