import os
import logging
import pandas as pd
import numpy as np
from scipy.stats import (
    pearsonr, spearmanr, kendalltau,
    f_oneway, kruskal, 
    ttest_ind, mannwhitneyu,
    shapiro, normaltest, levene
)
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import networkx as nx

class AnalysisEngine:
    """
    Advanced statistical analysis engine for Technology Posts
    
    DataOps Improvements:
    - Automatic selection of statistical tests based on data distribution.
    - Normality checks (Shapiro-Wilk / D'Agostino).
    - Homoscedasticity checks (Levene's Test).
    - Traceability of methods used in metadata.
    """

    def __init__(self, config):
        self.config = config
        self.analysis_config = config["analysis"]
        self.network_config = config.get("network_analysis", {"enabled": False})
        self.technology_domains = config["technology_posts"]["domains"]

    # =========================================================================
    # ESTADÍSTICA ROBUSTA 
    # =========================================================================

    def _check_normality(self, data):
        """
        Verifica si una serie de datos sigue una distribución normal.
        Retorna: (is_normal: bool, p_value: float)
        """
        clean_data = data.dropna()
        if len(clean_data) < 3:
            return False, 0.0
        
        try:
            # Para muestras grandes (>5000) shapiro es muy lento/inestable, usamos normaltest
            if len(clean_data) < 5000:
                stat, p_val = shapiro(clean_data)
            else:
                stat, p_val = normaltest(clean_data)
            
            # Si p > 0.05, NO rechazamos H0 -> Asumimos normalidad
            return p_val > 0.05, p_val
        except Exception as e:
            logging.warning(f"Normality test failed: {e}")
            return False, 0.0

    def _get_optimal_correlation(self, x, y, method_override=None):
        """
        Selecciona y calcula la correlación adecuada según la distribución de datos.
        """
        df_temp = pd.DataFrame({'x': x, 'y': y}).dropna()
        if len(df_temp) < 2:
            return None
            
        x_clean, y_clean = df_temp['x'], df_temp['y']
        
        # 1. Determinar método
        method_used = "pearson"
        is_x_normal, _ = self._check_normality(x_clean)
        is_y_normal, _ = self._check_normality(y_clean)

        if method_override:
            method_used = method_override
        elif not is_x_normal or not is_y_normal:
            # Si alguno no es normal, degradamos a Spearman (Rank)
            method_used = "spearman"
        
        # 2. Calcular
        try:
            if method_used == "pearson":
                corr, p_val = pearsonr(x_clean, y_clean)
            elif method_used == "spearman":
                corr, p_val = spearmanr(x_clean, y_clean)
            elif method_used == "kendall":
                corr, p_val = kendalltau(x_clean, y_clean)
            else:
                return None
                
            return {
                'correlation': float(corr),
                'p_value': float(p_val),
                'method_used': method_used,
                'significant': p_val < 0.05
            }
        except Exception as e:
            logging.error(f"Correlation calculation failed ({method_used}): {e}")
            return None

    # =========================================================================
    # MÓDULOS DE ANÁLISIS PRINCIPALES
    # =========================================================================

    def compute_correlations(self, df):
        """Compute multiple correlation analyses with statistical governance"""
        logging.info("Computing correlation analyses")
        correlations = {}
        
        # 1. Volume vs Engagement (Technology Posts)
        # Check config but also enforce logical availability of columns
        if self.analysis_config["correlations"].get("volume_vs_engagement", False):
            if 'reddit_engagement' in df.columns and 'word_count' in df.columns:
                result = self._get_optimal_correlation(df['word_count'], df['reddit_engagement'])
                if result:
                    correlations['volume_vs_engagement'] = result

        # 2. Influencer Activity vs Virality
        if self.analysis_config["correlations"].get("influencer_activity_vs_virality", False):
            if 'total_engagement' in df.columns and 'user_followers_count' in df.columns:
                result = self._get_optimal_correlation(df['user_followers_count'], df['total_engagement'])
                if result:
                    correlations['influencer_virality'] = result

        # 3. Sentiment vs Commercial Metrics (Generalized)
        if self.analysis_config["correlations"].get("sentiment_vs_commercial_metrics", False):
            sentiment_cols = [col for col in df.columns if 'vader_compound' in col or 'sentiment' in col]
            engagement_cols = [col for col in df.columns if 'engagement' in col.lower()]
            
            for sent_col in sentiment_cols:
                for eng_col in engagement_cols:
                    if pd.api.types.is_numeric_dtype(df[sent_col]) and pd.api.types.is_numeric_dtype(df[eng_col]):
                        result = self._get_optimal_correlation(df[sent_col], df[eng_col])
                        if result:
                            key = f"{sent_col}_vs_{eng_col}"
                            correlations[key] = result

        return correlations

    def perform_statistical_tests(self, df):
        """Perform ANOVA/Kruskal and T-test/Mann-Whitney"""
        logging.info("Performing statistical tests with variance checks")
        tests_results = {}
        
        # A. Comparison: Sentiment vs Engagement (ANOVA or Kruskal)
        if self.analysis_config["statistical_tests"].get("anova", False):
            if 'sentiment_label' in df.columns and 'total_engagement' in df.columns:
                try:
                    groups = []
                    group_labels = []
                    for sentiment in ['positive', 'neutral', 'negative']:
                        group_data = df[df['sentiment_label'] == sentiment]['total_engagement'].dropna()
                        if len(group_data) > 5: # Threshold mínimo
                            groups.append(group_data)
                            group_labels.append(sentiment)
                    
                    if len(groups) >= 2:
                        # 1. Test de Homocedasticidad (Levene)
                        # Si p < 0.05, varianzas son diferentes -> Usar test no paramétrico
                        stat_var, p_var = levene(*groups)
                        
                        if p_var < 0.05:
                            # Varianzas desiguales -> Kruskal-Wallis
                            stat, p_val = kruskal(*groups)
                            test_used = "Kruskal-Wallis (Heteroscedastic)"
                        else:
                            # Varianzas iguales -> ANOVA One-Way
                            stat, p_val = f_oneway(*groups)
                            test_used = "ANOVA (Fisher)"
                            
                        tests_results['sentiment_engagement_comparison'] = {
                            'statistic': float(stat),
                            'p_value': float(p_val),
                            'test_used': test_used,
                            'levene_p_value': float(p_var),
                            'significant': p_val < 0.05
                        }
                except Exception as e:
                    logging.warning(f"Group comparison failed: {e}")

        # B. Comparison: Platform Engagement (T-test or Mann-Whitney)
        if self.analysis_config["statistical_tests"].get("ttest", False):
            if 'data_source' in df.columns and 'engagement_rate' in df.columns:
                try:
                    twitter_eng = df[df['data_source'] == 'twitter']['engagement_rate'].dropna()
                    reddit_eng = df[df['data_source'].str.contains('reddit', case=False, na=False)]['engagement_rate'].dropna()
                    
                    if len(twitter_eng) > 5 and len(reddit_eng) > 5:
                        # Chequeo de normalidad para decidir test
                        is_norm_t, _ = self._check_normality(twitter_eng)
                        is_norm_r, _ = self._check_normality(reddit_eng)
                        
                        if is_norm_t and is_norm_r:
                            stat, p_val = ttest_ind(twitter_eng, reddit_eng, equal_var=False)
                            test_used = "Welch's T-test"
                        else:
                            stat, p_val = mannwhitneyu(twitter_eng, reddit_eng)
                            test_used = "Mann-Whitney U"
                            
                        tests_results['platform_engagement_comparison'] = {
                            'statistic': float(stat),
                            'p_value': float(p_val),
                            'test_used': test_used,
                            'significant': p_val < 0.05
                        }
                except Exception as e:
                    logging.warning(f"Platform comparison failed: {e}")

        return tests_results

    def perform_network_analysis(self, df):
        """Perform network analysis for influencer identification"""
        # Se mantiene idéntico a tu lógica original, solo agregando manejo de errores
        logging.info("Performing network analysis")
        
        if not self.network_config.get("enabled", False):
            return {}
        
        network_results = {}
        try:
            G = nx.Graph()
            
            # Construcción simple del grafo basada en usuarios (nodes)
            if 'user_id' in df.columns:
                users = df['user_id'].dropna().unique()
                # Aquí asumimos nodos aislados si no hay info de 'reply_to' o similar
                # Si tienes columnas de interacción, úsalas aquí para añadir edges
                for user in users:
                    G.add_node(user)
            
            # Si hay datos de interacciones (ej. reply_to_user_id), añadir edges
            if 'in_reply_to_user_id' in df.columns and 'user_id' in df.columns:
                interactions = df[['user_id', 'in_reply_to_user_id']].dropna()
                for _, row in interactions.iterrows():
                    G.add_edge(row['user_id'], row['in_reply_to_user_id'])

            if len(G.nodes()) > 0:
                network_results['network_metrics'] = {
                    'number_of_nodes': G.number_of_nodes(),
                    'number_of_edges': G.number_of_edges(),
                    'average_degree': np.mean([d for n, d in G.degree()]) if G.number_of_nodes() > 0 else 0
                }
                
                # Centralidad (si está configurada)
                if self.network_config.get("metrics"):
                    if "degree_centrality" in self.network_config["metrics"]:
                        degree_centrality = nx.degree_centrality(G)
                        # Top 10 centrality
                        network_results['degree_centrality'] = dict(sorted(
                            degree_centrality.items(), 
                            key=lambda x: x[1], 
                            reverse=True
                        )[:10]) 
                    
                    # Identificación de influencers por volumen (proxy si el grafo es muy disperso)
                    if 'user_id' in df.columns and 'total_engagement' in df.columns:
                        user_engagement = df.groupby('user_id')['total_engagement'].sum().sort_values(ascending=False)
                        network_results['top_influencers_by_engagement'] = user_engagement.head(10).to_dict()
                        
        except Exception as e:
            logging.error(f"Network analysis failed: {e}")
        
        return network_results

    def perform_time_series_analysis(self, df):
        """Perform time series analysis and decomposition"""
        logging.info("Performing time series analysis")
        time_series_results = {}
        
        try:
            date_cols = [col for col in df.columns if 'created' in col.lower() or 'date' in col.lower()]
            if not date_cols:
                return time_series_results
            
            date_col = date_cols[0]
            df_ts = df.copy()
            
            # Asegurar formato fecha
            df_ts[date_col] = pd.to_datetime(df_ts[date_col], errors='coerce')
            df_ts = df_ts.dropna(subset=[date_col]).sort_values(date_col)
            
            # Resampling diario
            numeric_cols = ['sentiment_polarity', 'total_engagement', 'reddit_engagement', 'word_count']
            agg_dict = {col: 'mean' if 'polarity' in col else 'sum' for col in numeric_cols if col in df_ts.columns}
            
            if not agg_dict:
                return {}

            daily_metrics = df_ts.set_index(date_col).resample('D').agg(agg_dict).dropna()
            
            if len(daily_metrics) < 7:  
                logging.info("Insufficient data points for Time Series Analysis (<7 days)")
                return time_series_results
            
            # Test de Estacionariedad (ADF)
            for column in daily_metrics.columns:
                try:
                    # ADF requiere variación en los datos
                    if daily_metrics[column].std() > 0:
                        adf_result = adfuller(daily_metrics[column])
                        time_series_results[f'adf_test_{column}'] = {
                            'test_statistic': float(adf_result[0]),
                            'p_value': float(adf_result[1]),
                            'is_stationary': adf_result[1] < 0.05
                        }
                except Exception as e:
                    logging.debug(f"ADF test failed for {column}: {e}")
            
            # Descomposición Estacional
            if self.config.get("timeseries", {}).get("decomposition", {}).get("apply", False):
                for column in ['sentiment_polarity', 'total_engagement']:
                    if column in daily_metrics.columns and len(daily_metrics[column]) >= 14: # Minimo 2 periodos
                        try:
                            decomposition = seasonal_decompose(
                                daily_metrics[column], 
                                model=self.config["timeseries"]["decomposition"].get("model", "additive"),
                                period=7
                            )
                            # Convertimos a lista para serialización JSON segura
                            time_series_results[f'decomposition_{column}'] = {
                                'trend': decomposition.trend.dropna().tolist(),
                                'seasonal': decomposition.seasonal.dropna().tolist(),
                                'residual': decomposition.resid.dropna().tolist()
                            }
                        except Exception as e:
                            logging.debug(f"Decomposition failed for {column}: {e}")
            
            # Resumen Estadístico de la Serie
            time_series_results['daily_metrics_summary'] = daily_metrics.describe().to_dict()
            
        except Exception as e:
            logging.error(f"Time series analysis failed: {e}")
        
        return time_series_results

    def perform_technology_domain_analysis(self, df):
        """Analyze technology domain-specific patterns"""
        logging.info("Analyzing technology domain patterns")
        domain_results = {}
        
        try:
            # Frecuencia de keywords
            if 'tech_keywords' in df.columns:
                # Asumiendo que tech_keywords es una lista o string
                # Si es string, tokenizar. Si es lista, expandir.
                pass # (Implementar si tienes la lógica de extracción de keywords lista)
            
            # Engagement por dominio (Technology Posts Requirement)
            domain_engagement = {}
            if 'text_cleaned' in df.columns:
                for domain in self.technology_domains:
                    # Búsqueda simple de substrings
                    mask = df['text_cleaned'].str.contains(domain.lower(), na=False)
                    domain_posts = df[mask]
                    
                    if len(domain_posts) > 0:
                        metrics = {
                            'count': int(len(domain_posts)),
                            'mean_engagement': float(domain_posts['total_engagement'].mean()) if 'total_engagement' in df.columns else 0,
                        }
                        # Agregar sentimiento si existe
                        if 'sentiment_polarity' in df.columns:
                            metrics['mean_sentiment'] = float(domain_posts['sentiment_polarity'].mean())
                            
                        domain_engagement[domain] = metrics
            
            domain_results['domain_engagement'] = domain_engagement
            
        except Exception as e:
            logging.error(f"Technology domain analysis failed: {e}")
        
        return domain_results

    def generate_cross_platform_insights(self, df):
        """Generate insights comparing different platforms"""
        logging.info("Generating cross-platform insights")
        cross_platform_results = {}
        
        try:
            if 'data_source' in df.columns:
                # Agrupación y descripción estadística
                aggs = {}
                if 'sentiment_polarity' in df.columns:
                    aggs['sentiment_polarity'] = ['mean', 'std', 'count']
                if 'total_engagement' in df.columns:
                    aggs['total_engagement'] = ['mean', 'sum', 'median'] # Mediana es mejor para distribuciones sesgadas
                
                if aggs:
                    platform_comparison = df.groupby('data_source').agg(aggs).round(4)
                    # Convertir multi-index columns a formato plano para JSON
                    flat_dict = {}
                    for col in platform_comparison.columns:
                        col_name = f"{col[0]}_{col[1]}"
                        flat_dict[col_name] = platform_comparison[col].to_dict()
                    
                    cross_platform_results['platform_summary'] = flat_dict
                
        except Exception as e:
            logging.error(f"Cross-platform analysis failed: {e}")
        
        return cross_platform_results

    def compute_project_specific_metrics(self, df):
        """
        Compute EXACT metrics required by the project (Technology Posts).
        NO HARDCODED VALUES - returns calculated metrics or explicit failures.
        """
        results = {}
        logging.info("Computing project specific metrics (Technology Posts)")
        
        try:
            # 1. Volume vs Engagement (Technology Posts Requirement)
            # Recalculamos usando el helper robusto
            if 'word_count' in df.columns and 'reddit_engagement' in df.columns:
                # Filtrar solo Reddit para esta métrica específica
                reddit_df = df[df['data_source'].str.contains('reddit', na=False)]
                if len(reddit_df) > 5:
                    corr_result = self._get_optimal_correlation(
                        reddit_df['word_count'], 
                        reddit_df['reddit_engagement']
                    )
                    if corr_result:
                        results['volume_vs_engagement'] = corr_result

            # 2. Cross-Platform Correlation (e.g. Trends in Twitter vs Reddit)
            # Esto requiere agrupar por tiempo
            date_cols = [col for col in df.columns if 'created' in col.lower() or 'date' in col.lower()]
            if date_cols and 'total_engagement' in df.columns:
                date_col = date_cols[0]
                # Pivotar: Index=Fecha, Columnas=Plataforma, Valores=Engagement Promedio
                pivot = df.pivot_table(
                    index=date_col, 
                    columns='data_source', 
                    values='total_engagement', 
                    aggfunc='mean'
                ).dropna()
                
                # Buscar correlación entre pares de plataformas disponibles
                if pivot.shape[1] >= 2:
                    cols = pivot.columns
                    # Ejemplo: correlacionar la primera con la segunda plataforma encontrada
                    c1, c2 = cols[0], cols[1]
                    corr_result = self._get_optimal_correlation(pivot[c1], pivot[c2])
                    if corr_result:
                        results['cross_platform_correlation'] = {
                            'platforms': f"{c1}_vs_{c2}",
                            **corr_result
                        }

        except Exception as e:
            logging.error(f"Project specific metrics computation failed: {e}")
        
        return results

    def generate_analysis_summary(self, analysis_results):
        """Generate a summary of key findings"""
        summary = {
            'status': 'completed',
            'counts': {
                'correlations': 0,
                'statistical_tests': 0,
                'domains_analyzed': 0
            },
            'significant_findings': []
        }
        
        try:
            # Contar correlaciones
            if 'correlations' in analysis_results:
                correlations = analysis_results['correlations']
                summary['counts']['correlations'] = len(correlations)
                
                # Identificar hallazgos significativos (p < 0.05)
                for name, res in correlations.items():
                    if res.get('significant', False):
                        summary['significant_findings'].append(
                            f"Correlation {name} is significant (p={res['p_value']:.4f}) using {res.get('method_used', 'unknown')}"
                        )
            
            # Contar tests
            if 'statistical_tests' in analysis_results:
                summary['counts']['statistical_tests'] = len(analysis_results['statistical_tests'])
            
            # Contar dominios
            if 'technology_domains' in analysis_results:
                domains = analysis_results['technology_domains'].get('domain_engagement', {})
                summary['counts']['domains_analyzed'] = len(domains)
            
        except Exception as e:
            logging.warning(f"Error generating analysis summary: {e}")
        
        return summary

    def run_analysis(self, df, network_results=None):
        """Execute complete analysis pipeline"""
        logging.info("Starting comprehensive statistical analysis")
        
        if not self.analysis_config["enabled"]:
            logging.info("Analysis disabled in configuration")
            return {}
            
        if df is None or df.empty:
            logging.error("Analysis aborted: Empty or None DataFrame provided")
            return {}
        
        try:
            analysis_results = {}
            
            # 1. Correlaciones (Robustas)
            analysis_results['correlations'] = self.compute_correlations(df)
            
            # 2. Tests de Hipótesis (ANOVA/Kruskal)
            analysis_results['statistical_tests'] = self.perform_statistical_tests(df)
            
            # 3. Análisis de Redes 
            analysis_results['network_analysis'] = network_results or self.perform_network_analysis(df)
            
            # 4. Series de Tiempo
            analysis_results['time_series'] = self.perform_time_series_analysis(df)
            
            # 5. Dominios Tecnológicos
            analysis_results['technology_domains'] = self.perform_technology_domain_analysis(df)
            
            # 6. Insights Multi-plataforma
            analysis_results['cross_platform'] = self.generate_cross_platform_insights(df)
            
            # 7. Métricas específicas del proyecto (Technology)
            analysis_results['project_specific_metrics'] = self.compute_project_specific_metrics(df)
            
            # 8. Resumen Ejecutivo
            analysis_results['summary'] = self.generate_analysis_summary(analysis_results)
            
            logging.info("Comprehensive analysis completed successfully")
            return analysis_results

        except Exception as e:
            logging.error(f"Analysis pipeline failed critical error: {e}")
            raise

    def run(self, df, network_results=None):
        return self.run_analysis(df, network_results)

def run_statistical_analysis(df, network_results, config):
    """Main entry point"""
    analyzer = AnalysisEngine(config)
    return analyzer.run(df, network_results)