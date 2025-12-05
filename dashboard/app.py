import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os
from datetime import datetime
from PIL import Image

# ==============================================================================
# 1. CONFIGURACION DE PAGINA
# ==============================================================================
st.set_page_config(
    page_title="Tech Trends Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS Corporativos
st.markdown("""
<style>
    .footer {
        position: fixed; left: 0; bottom: 0; width: 100%; 
        background-color: #f8f9fa; color: #6c757d; 
        text-align: center; padding: 10px; font-size: 12px; 
        border-top: 1px solid #dee2e6; z-index: 100;
    }
    .main-header {
        font-size: 2.2rem; font-weight: 600; color: #2c3e50; margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.1rem; color: #6c757d; margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. CARGA DE DATOS
# ==============================================================================
def get_run_versions(base_path="outputs"):
    """Retorna carpetas que contengan el archivo parquet procesado."""
    if not os.path.exists(base_path): return []
    valid_folders = []
    candidates = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    for folder in candidates:
        if os.path.exists(os.path.join(base_path, folder, "final_processed_data.parquet")):
            valid_folders.append(folder)
    return sorted(valid_folders, reverse=True)

@st.cache_data
def load_data(run_folder):
    """Carga dataset principal y reportes JSON."""
    data = {}
    
    # 1. Dataset Principal
    parquet_path = os.path.join(run_folder, "final_processed_data.parquet")
    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'], utc=True, errors='coerce')
        data['df'] = df
    
    # 2. Reportes JSON
    for json_file in ["statistical_analysis.json", "network_results.json", "nlp_analysis.json"]:
        path = os.path.join(run_folder, json_file)
        key = json_file.split('.')[0].replace('_results', '').replace('_analysis', '')
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data[key] = json.load(f)
    return data

# ==============================================================================
# 3. SIDEBAR (FILTROS)
# ==============================================================================
st.sidebar.title("Panel de Control")

versions = get_run_versions()
if not versions:
    st.error("No se encontraron datos procesados. Ejecuta el pipeline primero.")
    st.stop()

selected_version = st.sidebar.selectbox("Version de Ejecucion", versions, index=0)
run_path = os.path.join("outputs", selected_version)

data_bundle = load_data(run_path)
df = data_bundle.get('df')

if df is None:
    st.sidebar.error("Error critico: No se pudo cargar el dataset.")
    st.stop()

st.sidebar.divider()
st.sidebar.subheader("Filtros de Datos")

# 1. Plataforma
platforms = sorted(df['data_source'].unique())
selected_platforms = st.sidebar.multiselect("Fuente de Datos", platforms, default=platforms)

# 2. Fecha
df_dates = df.dropna(subset=['created_at'])
use_date_filter = False
date_range = []

if not df_dates.empty:
    min_date = df_dates['created_at'].min().date()
    max_date = df_dates['created_at'].max().date()
    
    st.sidebar.write("Rango de Fechas Disponible:")
    st.sidebar.caption(f"{min_date} al {max_date}")
    
    use_date_filter = st.sidebar.checkbox("Filtrar por rango de fechas", value=False)
    if use_date_filter:
        date_range = st.sidebar.date_input("Seleccionar Rango", [min_date, max_date])

# APLICAR FILTROS
mask = df['data_source'].isin(selected_platforms)
if use_date_filter and len(date_range) == 2:
    mask = mask & (df['created_at'].dt.date >= date_range[0]) & (df['created_at'].dt.date <= date_range[1])

df_filtered = df[mask]

# ==============================================================================
# 4. DASHBOARD PRINCIPAL
# ==============================================================================

st.markdown('<div class="main-header">Technology Trends Analytics</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">Analisis de tendencias en redes sociales | Run ID: {selected_version}</div>', unsafe_allow_html=True)
st.markdown("---")

# --- KPI SECTION ---
k1, k2, k3, k4 = st.columns(4)

with k1:
    k1.metric("Total Registros", f"{len(df_filtered):,}")

with k2:
    reddit_count = len(df_filtered[df_filtered['data_source'].str.contains('reddit', case=False)])
    k2.metric("Reddit (Kaggle)", f"{reddit_count:,}")

with k3:
    twitter_count = len(df_filtered[df_filtered['data_source'].str.contains('twitter', case=False)])
    k3.metric("Twitter (Simulado)", f"{twitter_count:,}")

with k4:
    eng = df_filtered['total_engagement'].sum() if 'total_engagement' in df_filtered.columns else 0
    k4.metric("Interacciones Totales", f"{eng:,.0f}")

st.markdown("---")

# --- TABS ---
tab_overview, tab_reddit, tab_stats, tab_nlp = st.tabs([
    "Vision General", 
    "Analisis Detallado Reddit", 
    "Estadistica", 
    "NLP y Temas"
])

# 1. VISION GENERAL
with tab_overview:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Volumen Diario por Fuente")
        if not df_filtered.empty and 'created_at' in df_filtered.columns:
            df_daily = df_filtered.groupby([pd.Grouper(key='created_at', freq='D'), 'data_source']).size().reset_index(name='count')
            fig = px.line(df_daily, x='created_at', y='count', color='data_source', markers=True)
            fig.update_layout(xaxis_title="Fecha", yaxis_title="Cantidad de Posts")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Datos insuficientes para grafico temporal.")
    
    with c2:
        st.subheader("Distribucion de Sentimiento")
        if 'sentiment_label' in df_filtered.columns:
            fig_pie = px.pie(
                df_filtered, 
                names='sentiment_label', 
                color='sentiment_label',
                color_discrete_map={'positive':'#2ecc71', 'negative':'#e74c3c', 'neutral':'#95a5a6'}
            )
            st.plotly_chart(fig_pie, use_container_width=True)

# 2. DEEP DIVE REDDIT
with tab_reddit:
    df_reddit = df_filtered[df_filtered['data_source'].str.contains('reddit', case=False)]
    
    if not df_reddit.empty:
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("#### Top Subreddits / Categorias")
            if 'subreddit' in df_reddit.columns:
                top_subs = df_reddit['subreddit'].value_counts().head(10).reset_index()
                top_subs.columns = ['Categoria', 'Posts']
                fig_bar = px.bar(top_subs, x='Posts', y='Categoria', orientation='h', color='Posts')
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.warning("Columna 'subreddit' no encontrada.")

        with c2:
            st.markdown("#### Distribucion de Engagement")
            fig_hist = px.histogram(df_reddit, x='total_engagement', nbins=30, title="Histograma de Interacciones")
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with st.expander("Ver Datos Crudos de Reddit (Top 50 por Engagement)"):
            cols_show = ['created_at', 'subreddit', 'text_cleaned', 'total_engagement', 'sentiment_label']
            cols_valid = [c for c in cols_show if c in df_reddit.columns]
            st.dataframe(
                df_reddit[cols_valid].sort_values('total_engagement', ascending=False).head(50),
                use_container_width=True
            )
    else:
        st.warning("No hay datos de Reddit para mostrar con los filtros actuales.")

# 3. ESTADISTICA (CORREGIDA)
with tab_stats:
    stats = data_bundle.get('statistical')
    if stats and 'correlations' in stats:
        st.subheader("Matriz de Correlaciones (Validacion de Hipotesis)")
        corr_list = []
        for k, v in stats['correlations'].items():
            
            # --- CORRECCIÓN AQUÍ: FORMATO CIENTÍFICO ---
            p_val = v.get('p_value', 1)
            
            if p_val == 0:
                p_text = "< 1e-100"  
            elif p_val < 0.0001:
                p_text = f"{p_val:.2e}"
            else:
                p_text = f"{p_val:.4f}"


            corr_list.append({
                "Hipotesis": k.replace('_', ' ').title(),
                "Coeficiente (r)": f"{v.get('correlation', 0):.3f}",
                "P-Value": p_text, # Usamos el texto formateado
                "Significativo": "SI" if v.get('significant') else "NO",
                "Metodo": v.get('method_used', '').upper()
            })
        st.dataframe(pd.DataFrame(corr_list), use_container_width=True)
    else:
        st.info("No se encontraron resultados estadisticos.")

    # Dispersión
    if 'sentiment_compound' in df_filtered.columns and 'total_engagement' in df_filtered.columns:
        st.markdown("#### Dispersion: Sentimiento vs Engagement")
        fig_scatter = px.scatter(
            df_filtered, 
            x="sentiment_compound", 
            y="total_engagement",
            color="data_source",
            size="word_count",
            hover_data=["text_cleaned"],
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

# 4. NLP & TOPICS
with tab_nlp:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Topicos Detectados (LDA)")
        nlp_data = data_bundle.get('nlp')
        if nlp_data and 'topics' in nlp_data:
            for t in nlp_data['topics']:
                st.code(t, language="text")
        else:
            st.warning("No hay topicos disponibles.")
            
    with c2:
        st.markdown("#### Largo de Texto vs Sentimiento")
        if not df_filtered.empty:
            fig_scat = px.scatter(
                df_filtered.sample(min(1000, len(df_filtered))), 
                x='word_count', 
                y='sentiment_compound', 
                color='data_source', 
                opacity=0.6,
                labels={'word_count': 'Palabras', 'sentiment_compound': 'Sentimiento (-1 a 1)'}
            )
            st.plotly_chart(fig_scat, use_container_width=True)

# FOOTER
st.markdown(f"""<div class="footer">Proyecto DataOps v2.0 | Pipeline Run: {selected_version} | Fecha: {datetime.now().strftime('%Y-%m-%d')}</div>""", unsafe_allow_html=True)