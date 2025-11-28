import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from datetime import datetime
import numpy as np

st.set_page_config(
    page_title="Technology Posts Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Technology Posts Analytics Dashboard")
st.markdown("Analisis de tendencias tecnologicas en redes sociales - **Proyecto Final DataOps**")

def find_processed_data():
    possible_paths = [
        "data/processed",
        "../data/processed", 
        "../../data/processed"
    ]
    
    for base_path in possible_paths:
        if os.path.exists(base_path):
            versions = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
            if versions:
                return base_path, versions
    return None, None

base_path, versions = find_processed_data()

if not versions:
    st.error("No se encontraron datos procesados. Ejecuta el pipeline primero.")
    st.info("Ejecuta: python run_pipeline.py --config config/pipeline_config.yaml")
    st.stop()

st.sidebar.title("Configuracion")
data_version = st.sidebar.selectbox(
    "Seleccionar version de datos",
    sorted(versions, reverse=True)
)

st.sidebar.header("Filtros")
min_engagement = st.sidebar.slider("Engagement minimo", 0, 1000, 0)
selected_platform = st.sidebar.multiselect(
    "Plataforma", 
    ["twitter", "reddit_kaggle", "reddit_api"],
    default=["twitter", "reddit_kaggle", "reddit_api"]
)

@st.cache_data
def load_processed_data(base_path, version):
    data_path = os.path.join(base_path, version, "processed_data.parquet")
    if os.path.exists(data_path):
        df = pd.read_parquet(data_path)
        
        # Asegurar metricas de engagement
        if 'reddit_engagement' not in df.columns and 'score' in df.columns and 'num_comments' in df.columns:
            df['reddit_engagement'] = df['score'] + (df['num_comments'] * 2)
        
        if 'total_engagement' not in df.columns:
            engagement_cols = [col for col in df.columns if 'engagement' in col.lower()]
            if engagement_cols:
                df['total_engagement'] = df[engagement_cols].sum(axis=1)
        
        return df
    return None

@st.cache_data
def load_analysis_results(base_path, version):
    analysis_path = os.path.join(base_path, version, "analysis_results.json")
    if os.path.exists(analysis_path):
        with open(analysis_path, 'r') as f:
            return json.load(f)
    return None

@st.cache_data
def load_network_analysis(base_path, version):
    network_path = os.path.join(base_path, version, "network_analysis.json")
    if os.path.exists(network_path):
        with open(network_path, 'r') as f:
            return json.load(f)
    return None

# Cargar datos
processed_data = load_processed_data(base_path, data_version)
analysis_data = load_analysis_results(base_path, data_version)
network_data = load_network_analysis(base_path, data_version)

if processed_data is None:
    st.error("No se pudieron cargar los datos procesados.")
    st.stop()

# Aplicar filtros
filtered_data = processed_data.copy()
if selected_platform:
    filtered_data = filtered_data[filtered_data['data_source'].isin(selected_platform)]
if min_engagement > 0 and 'total_engagement' in filtered_data.columns:
    filtered_data = filtered_data[filtered_data['total_engagement'] >= min_engagement]

# ============ SECCION 1: RESUMEN EJECUTIVO ============
st.header("Resumen Ejecutivo")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    total_posts = len(filtered_data)
    st.metric("Total de Posts", f"{total_posts:,}")

with col2:
    twitter_posts = len(filtered_data[filtered_data['data_source'] == 'twitter'])
    st.metric("Posts de Twitter", f"{twitter_posts:,}")

with col3:
    reddit_posts = len(filtered_data[filtered_data['data_source'].str.contains('reddit', na=False)])
    st.metric("Posts de Reddit", f"{reddit_posts:,}")

with col4:
    sentiment_cols = [col for col in filtered_data.columns if 'text_sentiment' in col and 'label' not in col and 'normalized' not in col]
    
    if not sentiment_cols:
        sentiment_cols = [col for col in filtered_data.columns if 'sentiment' in col and 'label' not in col and 'vader' not in col and 'normalized' not in col]
    
    if sentiment_cols:
        avg_sentiment = filtered_data[sentiment_cols[0]].mean()
        sentiment_color = "normal" if avg_sentiment > 0 else "off"
        st.metric("Sentimiento Promedio", f"{avg_sentiment:.3f}", delta_color=sentiment_color)
    else:
        st.metric("Sentimiento", "N/A")

with col5:
    if 'total_engagement' in filtered_data.columns:
        total_engagement = filtered_data['total_engagement'].sum()
        st.metric("Engagement Total", f"{total_engagement:,.0f}")
    else:
        st.metric("Engagement", "N/A")

# ============ SECCION 2: CORRELACIONES ESPECIFICAS DEL PROYECTO ============
st.header("Correlaciones Especificas del Proyecto")

if analysis_data and 'project_specific_metrics' in analysis_data:
    project_metrics = analysis_data['project_specific_metrics']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'volume_vs_engagement' in project_metrics:
            metric = project_metrics['volume_vs_engagement']
            corr = metric.get('pearson_correlation', 0)
            p_val = metric.get('p_value', 1)
            significant = metric.get('significant', False)
            
            status = "SIGNIFICATIVA" if significant else "NO SIGNIFICATIVA"
            
            st.metric(
                "Volumen vs Engagement Reddit", 
                f"{corr:.3f}",
                f"p={p_val:.3f} ({status})",
                delta_color="normal" if significant else "off"
            )
        else:
            st.metric("Volumen vs Engagement", "N/A", "Datos no disponibles")
    
    with col2:
        if 'influencer_activity_vs_virality' in project_metrics:
            metric = project_metrics['influencer_activity_vs_virality']
            corr = metric.get('pearson_correlation', 0)
            p_val = metric.get('p_value', 1)
            significant = metric.get('significant', False)
            
            status = "SIGNIFICATIVA" if significant else "NO SIGNIFICATIVA"
            
            st.metric(
                "Influencer Activity vs Virality", 
                f"{corr:.3f}",
                f"p={p_val:.3f} ({status})",
                delta_color="normal" if significant else "off"
            )
        else:
            st.metric("Influencer vs Virality", "N/A", "Datos no disponibles")
    
    with col3:
        if 'cross_platform_correlation' in project_metrics:
            metric = project_metrics['cross_platform_correlation']
            corr = metric.get('pearson_correlation', 0)
            p_val = metric.get('p_value', 1)
            significant = metric.get('significant', False)
            
            status = "SIGNIFICATIVA" if significant else "NO SIGNIFICATIVA"
            
            st.metric(
                "Correlacion Cruzada Plataformas", 
                f"{corr:.3f}",
                f"p={p_val:.3f} ({status})",
                delta_color="normal" if significant else "off"
            )
        else:
            st.metric("Correlacion Cruzada", "N/A", "Datos no disponibles")
else:
    st.warning("Las metricas especificas del proyecto no estan disponibles. Ejecuta el pipeline con el analisis actualizado.")

# ============ SECCION 3: ANALISIS DE PLATAFORMAS ============
st.header("Analisis por Plataforma")

col1, col2 = st.columns(2)

with col1:
    platform_dist = filtered_data['data_source'].value_counts()
    if len(platform_dist) > 0:
        fig_platform = px.pie(
            values=platform_dist.values,
            names=platform_dist.index,
            title='Distribucion de Posts por Plataforma',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_platform, use_container_width=True)
    else:
        st.info("No hay datos de plataformas para mostrar")

with col2:
    if 'total_engagement' in filtered_data.columns and len(filtered_data) > 0:
        engagement_by_platform = filtered_data.groupby('data_source')['total_engagement'].mean().reset_index()
        if len(engagement_by_platform) > 0:
            fig_engagement_platform = px.bar(
                engagement_by_platform,
                x='data_source',
                y='total_engagement',
                title='Engagement Promedio por Plataforma',
                color='data_source',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_engagement_platform.update_layout(xaxis_title="Plataforma", yaxis_title="Engagement Promedio")
            st.plotly_chart(fig_engagement_platform, use_container_width=True)
        else:
            st.info("No hay datos de engagement por plataforma")
    else:
        st.info("No hay datos de engagement disponibles")

# ============ SECCION 4: ANALISIS DE SENTIMIENTO ============
st.header("Analisis de Sentimiento")

if len(filtered_data) > 0:
    col1, col2 = st.columns(2)

    with col1:
        sentiment_cols = [col for col in filtered_data.columns if 'text_sentiment' in col and 'label' not in col and 'normalized' not in col]
        if sentiment_cols:
            sentiment_data = filtered_data.dropna(subset=[sentiment_cols[0]])
            
            if len(sentiment_data) > 0:
                fig_sentiment = px.histogram(
                    sentiment_data, 
                    x=sentiment_cols[0],
                    title=f'Distribución de Sentimiento ({len(sentiment_data):,} registros)',
                    nbins=20,
                    color_discrete_sequence=['#FF6B6B'],
                    opacity=0.7
                )
                fig_sentiment.update_layout(
                    xaxis_title="Puntuación de Sentimiento", 
                    yaxis_title="Frecuencia",
                    xaxis_range=[-1, 1]
                )
                st.plotly_chart(fig_sentiment, use_container_width=True)
                
                st.metric("Registros con sentimiento", f"{len(sentiment_data):,}")
                st.metric("Media de sentimiento", f"{sentiment_data[sentiment_cols[0]].mean():.3f}")
            else:
                st.info("No hay datos de sentimiento para mostrar")
        else:
            st.info("No se encontró la columna text_sentiment")

    with col2:
        st.subheader("Distribución de Sentimientos por Plataforma")
        
        sentiment_dist = filtered_data.groupby(['data_source', 'text_sentiment_label']).size().unstack(fill_value=0)
        
        fig_dist = px.bar(
            sentiment_dist,
            x=sentiment_dist.index,
            y=sentiment_dist.columns,
            title='Distribución de Sentimientos por Plataforma',
            color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1'],  # neg, pos, neutral
            barmode='stack'
        )
        fig_dist.update_layout(
            xaxis_title="Plataforma",
            yaxis_title="Cantidad de Posts",
            showlegend=True
        )
        st.plotly_chart(fig_dist, use_container_width=True)
        
        st.metric("Total Neutral", f"{sentiment_dist['neutral'].sum():,}")
        st.metric("Total Positive", f"{sentiment_dist['positive'].sum():,}") 
        st.metric("Total Negative", f"{sentiment_dist['negative'].sum():,}")
        
else:
    st.info("No hay datos disponibles para analisis de sentimiento")

# ============ SECCION 5: ANALISIS DE ENGAGEMENT ============
st.header("Analisis de Engagement")

if len(filtered_data) > 0:
    col1, col2 = st.columns(2)

    with col1:
        engagement_cols = ['total_engagement', 'reddit_engagement', 'twitter_engagement']
        available_engagement_cols = [col for col in engagement_cols if col in filtered_data.columns]
        
        if available_engagement_cols:
            engagement_col = available_engagement_cols[0]  
            sample_for_top = filtered_data.nlargest(1000, engagement_col)
            top_engagement = sample_for_top.nlargest(min(10, len(sample_for_top)), engagement_col)
            
            if len(top_engagement) > 0:
                text_cols = [col for col in top_engagement.columns if any(x in col for x in ['text', 'title', 'selftext']) and 'cleaned' in col]
                display_col = text_cols[0] if text_cols else top_engagement.columns[0]
                
                top_engagement['display_text'] = top_engagement[display_col].astype(str).str[:80] + "..."
                
                fig_engagement = px.bar(
                    top_engagement,
                    x=engagement_col,
                    y='display_text',
                    title=f'Top {len(top_engagement)} Posts por Engagement ({engagement_col})',
                    color='data_source',
                    orientation='h',
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_engagement.update_layout(
                    yaxis_title="Post", 
                    xaxis_title=f"Engagement ({engagement_col})",
                    showlegend=True
                )
                st.plotly_chart(fig_engagement, use_container_width=True)
                
                st.metric(f"Engagement máximo ({engagement_col})", f"{top_engagement[engagement_col].max():,.0f}")
                st.metric(f"Engagement promedio", f"{filtered_data[engagement_col].mean():,.0f}")
            else:
                st.info("No hay posts con engagement suficiente")
        else:
            st.info("No se encontraron metricas de engagement")

    with col2:
        sentiment_col = 'text_sentiment' if 'text_sentiment' in filtered_data.columns else None
        engagement_col = 'total_engagement' if 'total_engagement' in filtered_data.columns else None
        
        if sentiment_col and engagement_col and len(filtered_data) >= 2:
            sample_size = min(2000, len(filtered_data))
            sample_data = filtered_data.sample(n=sample_size) if len(filtered_data) > sample_size else filtered_data
            
            fig_correlation = px.scatter(
                sample_data,
                x=sentiment_col,
                y=engagement_col,
                title=f'Correlacion: Sentimiento vs Engagement (muestra de {len(sample_data)})',
                color='data_source',
                trendline="lowess",
                color_discrete_sequence=px.colors.qualitative.Set3,
                opacity=0.6
            )
            fig_correlation.update_layout(
                xaxis_title="Sentimiento (text_sentiment)", 
                yaxis_title=f"Engagement ({engagement_col})",
                xaxis_range=[-1, 1]
            )
            st.plotly_chart(fig_correlation, use_container_width=True)
            
            correlation = sample_data[sentiment_col].corr(sample_data[engagement_col])
            st.metric("Correlacion Sentimiento-Engagement", f"{correlation:.3f}")
            
        else:
            st.info("No hay suficientes datos para correlacion")
            missing_cols = []
            if 'text_sentiment' not in filtered_data.columns:
                missing_cols.append('text_sentiment')
            if 'total_engagement' not in filtered_data.columns:
                missing_cols.append('total_engagement')
            if missing_cols:
                st.error(f"Columnas faltantes: {missing_cols}")
else:
    st.info("No hay datos disponibles para analisis de engagement")

# ============ SECCION 6: ANALISIS TEMPORAL ============
st.header("Analisis Temporal")

if len(filtered_data) > 0:
    real_data = filtered_data[filtered_data['data_source'].str.contains('reddit_kaggle', na=False)]
    
    if len(real_data) > 0:
        date_priority = ['created_utc', 'datetime', 'created_at', 'date']
        date_cols = [col for col in date_priority if col in real_data.columns]
        
        if date_cols:
            try:
                date_col = date_cols[0]
                
                if date_col == 'created_utc':
                    real_data[date_col] = pd.to_datetime(real_data[date_col], unit='s', errors='coerce')
                else:
                    real_data[date_col] = pd.to_datetime(real_data[date_col], errors='coerce')
                
                temporal_data = real_data.dropna(subset=[date_col])
                
                if len(temporal_data) > 0:
                    min_date = temporal_data[date_col].min()
                    max_date = temporal_data[date_col].max()
                    date_range_days = (max_date - min_date).days
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Fecha inicial", min_date.strftime('%d/%m/%Y'))
                    with col2:
                        st.metric("Fecha final", max_date.strftime('%d/%m/%Y'))
                    with col3:
                        st.metric("Período analizado", f"{date_range_days} días")
                    
                    if date_range_days > 730:  
                        temporal_data['time_period'] = temporal_data[date_col].dt.to_period('Q').astype(str)
                        period_label = "Trimestre"
                        points = len(temporal_data['time_period'].unique())
                    elif date_range_days > 180:  
                        temporal_data['time_period'] = temporal_data[date_col].dt.to_period('M').astype(str)
                        period_label = "Mes"
                        points = len(temporal_data['time_period'].unique())
                    elif date_range_days > 60: 
                        temporal_data['time_period'] = temporal_data[date_col].dt.to_period('W').astype(str)
                        period_label = "Semana"
                        points = len(temporal_data['time_period'].unique())
                    else: 
                        temporal_data['time_period'] = temporal_data[date_col].dt.date.astype(str)
                        period_label = "Día"
                        points = len(temporal_data['time_period'].unique())
                    
                    
                    # ===== VOLUMEN DE POSTS =====
                    posts_per_period = temporal_data.groupby('time_period').size().reset_index()
                    posts_per_period.columns = ['period', 'post_count']
                    
                    if len(posts_per_period) > 1:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig_volume = px.line(
                                posts_per_period,
                                x='period',
                                y='post_count',
                                title=f'Volumen de Posts por {period_label}',
                                markers=True,
                                line_shape='spline'
                            )
                            fig_volume.update_layout(
                                xaxis_title=period_label, 
                                yaxis_title="Número de Posts",
                                xaxis_tickangle=45
                            )
                            st.plotly_chart(fig_volume, use_container_width=True)
                        
                        with col2:
                            fig_volume_bar = px.bar(
                                posts_per_period.tail(20), 
                                x='period',
                                y='post_count',
                                title=f'Volumen de Posts (Últimos 20 {period_label}s)',
                                color='post_count',
                                color_continuous_scale='Viridis'
                            )
                            fig_volume_bar.update_layout(
                                xaxis_title=period_label,
                                yaxis_title="Número de Posts",
                                xaxis_tickangle=45
                            )
                            st.plotly_chart(fig_volume_bar, use_container_width=True)
                    
                    # ===== ENGAGEMENT TEMPORAL =====
                    st.subheader("Análisis de Engagement Temporal")
                    
                    engagement_period = temporal_data.groupby('time_period').agg({
                        'total_engagement': ['sum', 'mean', 'max', 'count']
                    }).round(2)
                    
                    engagement_period.columns = ['engagement_total', 'engagement_promedio', 'engagement_maximo', 'post_count']
                    engagement_period = engagement_period.reset_index()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_engagement_total = px.line(
                            engagement_period,
                            x='time_period',
                            y='engagement_total',
                            title=f'Engagement Total por {period_label}',
                            markers=True,
                            line_shape='spline'
                        )
                        fig_engagement_total.update_layout(
                            xaxis_title=period_label,
                            yaxis_title="Engagement Total",
                            xaxis_tickangle=45
                        )
                        st.plotly_chart(fig_engagement_total, use_container_width=True)
                    
                    with col2:
                        fig_engagement_avg = px.line(
                            engagement_period,
                            x='time_period',
                            y='engagement_promedio',
                            title=f'Engagement Promedio por {period_label}',
                            markers=True,
                            line_shape='spline',
                            color_discrete_sequence=['#FF6B6B']
                        )
                        fig_engagement_avg.update_layout(
                            xaxis_title=period_label,
                            yaxis_title="Engagement Promedio",
                            xaxis_tickangle=45
                        )
                        st.plotly_chart(fig_engagement_avg, use_container_width=True)
                    
                    # ===== MÉTRICAS RESUMEN TEMPORALES =====
                    st.subheader("Métricas Temporales Clave")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        peak_period = posts_per_period.loc[posts_per_period['post_count'].idxmax()]
                        st.metric(f"Mayor volumen ({period_label})", 
                                 f"{peak_period['post_count']} posts", 
                                 f"{peak_period['period']}")
                    
                    with col2:
                        peak_engagement = engagement_period.loc[engagement_period['engagement_total'].idxmax()]
                        st.metric(f"Mayor engagement ({period_label})", 
                                 f"{peak_engagement['engagement_total']:,.0f}",
                                 f"{peak_engagement['time_period']}")
                    
                    with col3:
                        avg_posts_per_period = posts_per_period['post_count'].mean()
                        st.metric("Promedio posts/periodo", f"{avg_posts_per_period:.1f}")
                    
                    with col4:
                        engagement_growth = ((engagement_period['engagement_total'].iloc[-1] - 
                                           engagement_period['engagement_total'].iloc[0]) / 
                                           engagement_period['engagement_total'].iloc[0] * 100)

                        
                else:
                    st.info("No hay datos temporales válidos después de la conversión")
                    
            except Exception as e:
                st.warning(f"No se pudieron procesar las fechas: {e}")
        else:
            st.info("No se encontraron columnas de fecha en los datos reales")
    else:
        st.warning("No hay datos REALES de Reddit para análisis temporal")
else:
    st.info("No hay datos disponibles para análisis temporal")


# ============ SECCION 7: SISTEMA DE ALERTAS ============
st.header("Sistema de Alertas")

# ALERTAS CALCULADAS EN TIEMPO REAL DESDE LOS DATOS
alerts_triggered = []

# 1. Alerta: Caída de Sentimiento
if 'text_sentiment' in filtered_data.columns:
    current_sentiment = filtered_data['text_sentiment'].mean()
    sentiment_alert_threshold = -0.1
    
    if current_sentiment < sentiment_alert_threshold:
        alerts_triggered.append({
            'type': 'error',
            'message': f'Caída de Sentimiento - Actual: {current_sentiment:.3f} (Límite: {sentiment_alert_threshold})',
            'metric': 'Sentimiento Promedio'
        })

# 2. Alerta: Engagement Muy Bajo
if 'total_engagement' in filtered_data.columns:
    avg_engagement = filtered_data['total_engagement'].mean()
    engagement_alert_threshold = 10
    
    if avg_engagement < engagement_alert_threshold:
        alerts_triggered.append({
            'type': 'warning', 
            'message': f'Engagement Promedio Bajo - Actual: {avg_engagement:.1f} (Mínimo: {engagement_alert_threshold})',
            'metric': 'Engagement Promedio'
        })

# 3. Alerta: Desbalance de Plataformas
platform_counts = filtered_data['data_source'].value_counts()
if len(platform_counts) > 0:
    max_platform = platform_counts.max()
    min_platform = platform_counts.min()
    balance_ratio = min_platform / max_platform if max_platform > 0 else 0
    
    if balance_ratio < 0.3:  # Si una plataforma tiene menos del 30% de la mayor
        alerts_triggered.append({
            'type': 'info',
            'message': f'Desbalance de Plataformas - Ratio: {balance_ratio:.2f}',
            'metric': 'Balance Plataformas'
        })

# 4. Alerta: Correlación Baja (si existe analysis_data)
if analysis_data and 'project_specific_metrics' in analysis_data:
    metrics = analysis_data['project_specific_metrics']
    if 'volume_vs_engagement' in metrics:
        corr = metrics['volume_vs_engagement'].get('pearson_correlation', 0)
        if abs(corr) < 0.3:  # Correlación muy débil
            alerts_triggered.append({
                'type': 'warning',
                'message': f'Correlación Volumen-Engagement Débil: {corr:.3f}',
                'metric': 'Correlación Principal'
            })

# MOSTRAR ALERTAS
if alerts_triggered:
    st.subheader("Alertas Activas")
    
    for alert in alerts_triggered:
        if alert['type'] == 'error':
            st.error(f"**{alert['metric']}** - {alert['message']}")
        elif alert['type'] == 'warning':
            st.warning(f"**{alert['metric']}** - {alert['message']}")
        else:
            st.info(f"**{alert['metric']}** - {alert['message']}")
else:
    st.success("Todas las métricas dentro de rangos normales")

# RESUMEN DE ESTADO
st.subheader("Estado del Sistema")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if 'text_sentiment' in filtered_data.columns:
        current_sentiment = filtered_data['text_sentiment'].mean()
        status = "OK" if current_sentiment >= -0.1 else "BAD"
        st.metric("Sentimiento", f"{current_sentiment:.3f}", status)

with col2:
    if 'total_engagement' in filtered_data.columns:
        avg_engagement = filtered_data['total_engagement'].mean()
        status = "OK" if avg_engagement >= 10 else "LOW"
        st.metric("Engagement Avg", f"{avg_engagement:.1f}", status)

with col3:
    platform_counts = filtered_data['data_source'].value_counts()
    if len(platform_counts) > 0:
        balance_ratio = platform_counts.min() / platform_counts.max() if platform_counts.max() > 0 else 0
        status = "OK" if balance_ratio >= 0.3 else "LOW"
        st.metric("Balance", f"{balance_ratio:.2f}", status)

with col4:
    total_alerts = len(alerts_triggered)
    status = "OK" if total_alerts == 0 else "ACTIVE"
    st.metric("Alertas Activas", total_alerts, status)


# ============ SECCION 8: EXPLORAR DATOS ============
st.header("Explorar Datos")

tab1, tab2, tab3 = st.tabs(["Datos Sample", "Estadisticas", "Busqueda"])

with tab1:
    st.subheader("Muestra de Datos Procesados")
    if len(filtered_data) > 0:
        sample_size = st.slider("Tamaño de muestra", 100, 5000, min(1000, len(filtered_data)))
        sample_data = filtered_data.sample(n=min(sample_size, len(filtered_data)))
        
        st.write(f"**Muestra de {len(sample_data)} registros** (de {len(filtered_data)} totales)")
        
        important_cols = ['data_source', 'text_cleaned', 'title_cleaned', 'total_engagement', 
                         'text_sentiment', 'text_sentiment_label', 'dominant_topic']
        display_cols = [col for col in important_cols if col in sample_data.columns]
        
        if not display_cols:
            display_cols = sample_data.columns[:8].tolist()
        
        st.dataframe(sample_data[display_cols], use_container_width=True)
    else:
        st.info("No hay datos para mostrar")

with tab2:
    st.subheader("Estadisticas Descriptivas")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Resumen por Plataforma:**")
        platform_stats = filtered_data['data_source'].value_counts()
        if len(platform_stats) > 0:
            total_posts = len(filtered_data)
            for platform, count in platform_stats.items():
                percentage = (count / total_posts) * 100
                st.write(f"- **{platform}**: {count:,} posts ({percentage:.1f}%)")
        else:
            st.write("No hay datos de plataformas")
    
    with col2:
        st.write("**Metricas:**")

        priority_numeric = [
            'total_engagement',
            'text_sentiment',
            'word_count',
            'score',
            'comments'
        ]

        numeric_cols = [col for col in priority_numeric if col in filtered_data.columns]
        
        if not numeric_cols:
            numeric_cols = filtered_data.select_dtypes(include=[np.number]).columns.tolist()[:5]
        
        if len(numeric_cols) > 0:
            for col in numeric_cols:
                if col in filtered_data.columns:
                    mean_val = filtered_data[col].mean()
                    std_val = filtered_data[col].std()
                    st.write(f"- **{col}**: {mean_val:.2f} ± {std_val:.2f}")
        else:
            st.write("No hay metricas numericas")


with tab3:
    st.subheader("Busqueda en Posts")
    search_term = st.text_input("Buscar en textos:")
    if search_term:
        text_cols = [col for col in filtered_data.columns if any(x in col for x in ['text', 'title', 'selftext']) and 'cleaned' in col]
        if text_cols and len(filtered_data) > 0:
            mask = filtered_data[text_cols[0]].str.contains(search_term, case=False, na=False)
            results = filtered_data[mask]
            st.write(f"**{len(results)}** posts encontrados con '{search_term}'")
            if len(results) > 0:
                display_cols = ['data_source', text_cols[0], 'total_engagement', 'text_sentiment_label']
                display_cols = [col for col in display_cols if col in results.columns]
                
                st.dataframe(results[display_cols].head(10), use_container_width=True)
                
                if len(results) > 5:
                    st.write("**Distribución por plataforma:**")
                    platform_dist = results['data_source'].value_counts()
                    for platform, count in platform_dist.items():
                        st.write(f"- {platform}: {count} posts")
            else:
                st.info("No se encontraron posts con el termino de busqueda")
        else:
            st.info("No hay datos para buscar")
    else:
        st.info("Ingresa un termino de busqueda para explorar los posts")

# ============ FOOTER ============
st.markdown("---")
st.markdown(
    "**Technology Posts Pipeline** | "
    f"Ultima actualizacion: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
    f"Version de datos: {data_version} | "
    f"Total registros: {len(filtered_data):,} | "
    f"Filtrados: {len(filtered_data):,} de {len(processed_data):,}"
)