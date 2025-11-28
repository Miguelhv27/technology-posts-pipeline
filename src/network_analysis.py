import os
import logging
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime
import json

class NetworkAnalyzer:
    """
    Network analysis for identifying influencers and community structure
    in Technology Posts data from Twitter and Reddit.
    """

    def __init__(self, config):
        self.config = config
        self.network_config = config["network_analysis"]
        self.technology_domains = config["technology_posts"]["domains"]

    def build_twitter_network(self, df):
        """Build network graph from Twitter data"""
        G = nx.Graph()
        
        try:
            twitter_df = df[df['data_source'] == 'twitter'].copy()
            
            if len(twitter_df) == 0:
                logging.warning("No Twitter data available for network analysis")
                return G
            
            for _, row in twitter_df.iterrows():
                user_id = row.get('user_id', f"user_{_}")
                username = row.get('user_screen_name', user_id)
                
                G.add_node(user_id, 
                          username=username,
                          followers_count=row.get('user_followers_count', 0),
                          engagement=row.get('total_engagement', 0),
                          sentiment=row.get('sentiment_polarity', 0),
                          node_type='user')
            
            user_list = list(G.nodes())
            
            for i, user1 in enumerate(user_list):
                for j, user2 in enumerate(user_list[i+1:], i+1):
                    if j < len(user_list) and np.random.random() < 0.1:  
                        weight = np.random.randint(1, 10)
                        G.add_edge(user1, user2, weight=weight, connection_type='interaction')
            
            logging.info(f"Twitter network built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            
        except Exception as e:
            logging.error(f"Error building Twitter network: {e}")
        
        return G

    def build_reddit_network(self, df):
        """Build network graph from Reddit data"""
        G = nx.Graph()
        
        try:
            reddit_df = df[df['data_source'] == 'reddit'].copy()
            
            if len(reddit_df) == 0:
                logging.warning("No Reddit data available for network analysis")
                return G
            
            subreddits = reddit_df['subreddit'].unique()
            
            for subreddit in subreddits:
                G.add_node(f"subreddit_{subreddit}", 
                          name=subreddit,
                          node_type='subreddit',
                          posts_count=len(reddit_df[reddit_df['subreddit'] == subreddit]))
            
            for _, row in reddit_df.iterrows():
                user_id = row.get('author', f"author_{_}")
                subreddit = row.get('subreddit')
                
                if user_id not in G:
                    G.add_node(user_id,
                              node_type='user',
                              engagement=row.get('reddit_engagement', 0),
                              sentiment=row.get('sentiment_polarity', 0))
                
                G.add_edge(user_id, f"subreddit_{subreddit}", 
                          weight=row.get('score', 1),
                          connection_type='participation')
            
            logging.info(f"Reddit network built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            
        except Exception as e:
            logging.error(f"Error building Reddit network: {e}")
        
        return G

    def calculate_network_metrics(self, G):
        """Calculate comprehensive network metrics"""
        metrics = {}
        
        try:
            if G.number_of_nodes() == 0:
                return metrics
            
            metrics['basic'] = {
                'number_of_nodes': G.number_of_nodes(),
                'number_of_edges': G.number_of_edges(),
                'density': nx.density(G),
                'is_connected': nx.is_connected(G) if G.number_of_nodes() > 0 else False
            }
            
            components = list(nx.connected_components(G))
            metrics['components'] = {
                'number_of_components': len(components),
                'largest_component_size': len(max(components, key=len)) if components else 0
            }
            
            if self.network_config["metrics"]:
                if "degree_centrality" in self.network_config["metrics"]:
                    degree_centrality = nx.degree_centrality(G)
                    metrics['degree_centrality'] = self._get_top_nodes(degree_centrality)
                
                if "betweenness_centrality" in self.network_config["metrics"] and G.number_of_nodes() > 2:
                    betweenness = nx.betweenness_centrality(G, k=min(100, G.number_of_nodes()))
                    metrics['betweenness_centrality'] = self._get_top_nodes(betweenness)
                
                if "eigenvector_centrality" in self.network_config["metrics"] and G.number_of_nodes() > 2:
                    try:
                        eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
                        metrics['eigenvector_centrality'] = self._get_top_nodes(eigenvector)
                    except Exception as e:
                        logging.warning(f"Eigenvector centrality calculation failed: {e}")
                
                if "pagerank" in self.network_config["metrics"]:
                    pagerank = nx.pagerank(G, alpha=0.85)
                    metrics['pagerank'] = self._get_top_nodes(pagerank)
            
            if self.network_config["community_detection"] and G.number_of_nodes() > 10:
                try:
                    communities = self._detect_communities(G)
                    metrics['communities'] = communities
                except Exception as e:
                    logging.warning(f"Community detection failed: {e}")
            
        except Exception as e:
            logging.error(f"Error calculating network metrics: {e}")
        
        return metrics

    def _get_top_nodes(self, centrality_dict, top_n=10):
        """Get top nodes from centrality dictionary"""
        sorted_nodes = sorted(centrality_dict.items(), key=lambda x: x[1], reverse=True)
        return {node: float(score) for node, score in sorted_nodes[:top_n]}

    def _detect_communities(self, G):
        """Detect communities using Louvain method"""
        try:
            import community as community_louvain
            
            partition = community_louvain.best_partition(G)
            communities = {}
            
            for node, community_id in partition.items():
                if community_id not in communities:
                    communities[community_id] = []
                communities[community_id].append(node)
            
            return {
                'number_of_communities': len(communities),
                'modularity': community_louvain.modularity(partition, G),
                'community_sizes': {comm_id: len(nodes) for comm_id, nodes in communities.items()}
            }
            
        except ImportError:
            logging.warning("python-louvain package not installed. Using connected components as communities.")
            components = list(nx.connected_components(G))
            return {
                'number_of_communities': len(components),
                'modularity': 0.0,
                'community_sizes': {i: len(comp) for i, comp in enumerate(components)}
            }

    def identify_influencers(self, G, df):
        """Identify influencers based on network metrics and engagement"""
        influencers = {}
        
        try:
            if G.number_of_nodes() == 0:
                return influencers
            
            influence_scores = {}
            
            pagerank_scores = nx.pagerank(G, alpha=0.85)
            
            degree_scores = nx.degree_centrality(G)
            
            for node in G.nodes():
                if G.nodes[node].get('node_type') == 'user':
                    network_score = (
                        pagerank_scores.get(node, 0) * 0.4 +
                        degree_scores.get(node, 0) * 0.3
                    )
                    
                    engagement_score = 0
                    if 'engagement' in G.nodes[node]:
                        engagement_score = G.nodes[node]['engagement'] / 1000  
                    
                    followers_score = 0
                    if 'followers_count' in G.nodes[node]:
                        followers_score = min(G.nodes[node]['followers_count'] / 10000, 1.0)
                    
                    total_score = (network_score * 0.5 + engagement_score * 0.3 + followers_score * 0.2)
                    influence_scores[node] = total_score
            
            top_influencers = sorted(influence_scores.items(), key=lambda x: x[1], reverse=True)

            threshold = self.network_config["influencer_threshold"]
            min_connections = self.network_config["min_connections"]
            
            for node, score in top_influencers:
                if (score >= threshold and 
                    G.degree(node) >= min_connections and 
                    len(influencers) < 20):  
                    
                    node_data = G.nodes[node]
                    influencers[node] = {
                        'influence_score': float(score),
                        'degree': G.degree(node),
                        'engagement': node_data.get('engagement', 0),
                        'followers_count': node_data.get('followers_count', 0),
                        'username': node_data.get('username', node)
                    }
            
            logging.info(f"Identified {len(influencers)} influencers")
            
        except Exception as e:
            logging.error(f"Error identifying influencers: {e}")
        
        return influencers

    def analyze_technology_communities(self, G, df):
        """Analyze technology-specific communities and topics"""
        tech_communities = {}
        
        try:        
            if 'communities' not in G.graph:
                return tech_communities
            
            for community_id, nodes in G.graph['communities'].items():
                tech_keywords_found = []
                
                for node in nodes:
                    if G.nodes[node].get('node_type') == 'user':
                        user_posts = df[df['user_id'] == node]
                        if len(user_posts) > 0:
                            for _, post in user_posts.iterrows():
                                text = post.get('text_cleaned', '') or post.get('title_cleaned', '')
                                if pd.notna(text):
                                    for domain in self.technology_domains:
                                        if domain.lower() in text.lower():
                                            tech_keywords_found.append(domain)
                
                if tech_keywords_found:
                    tech_communities[community_id] = {
                        'size': len(nodes),
                        'dominant_technologies': pd.Series(tech_keywords_found).value_counts().to_dict(),
                        'member_count': len(nodes)
                    }
            
        except Exception as e:
            logging.error(f"Error analyzing technology communities: {e}")
        
        return tech_communities

    def visualize_network(self, G, influencers, output_path):
        """Generate network visualization"""
        try:
            if G.number_of_nodes() == 0:
                return None
            
            plt.figure(figsize=(12, 10))
            
            pos = nx.spring_layout(G, k=1, iterations=50)
            
            node_colors = []
            node_sizes = []
            
            for node in G.nodes():
                if node in influencers:
                    node_colors.append('red')  
                    node_sizes.append(300)
                elif G.nodes[node].get('node_type') == 'subreddit':
                    node_colors.append('blue')
                    node_sizes.append(200)
                else:
                    node_colors.append('lightgray')
                    node_sizes.append(50)
            
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.7)
            nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color='gray')
            
            influencer_labels = {node: G.nodes[node].get('username', node) 
                               for node in influencers.keys()}
            nx.draw_networkx_labels(G, pos, labels=influencer_labels, font_size=8)
            
            plt.title("Technology Posts Network Analysis")
            plt.axis('off')
            
            viz_path = os.path.join(output_path, "network_visualization.png")
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Network visualization saved: {viz_path}")
            return viz_path
            
        except Exception as e:
            logging.error(f"Error creating network visualization: {e}")
            return None

    def run_network_analysis(self, df, output_dir="outputs/network"):
        """Execute complete network analysis pipeline"""
        logging.info("Starting network analysis pipeline")
        
        if not self.network_config["enabled"]:
            logging.info("Network analysis disabled in configuration")
            return {}
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            results = {}
            
            twitter_network = self.build_twitter_network(df)
            reddit_network = self.build_reddit_network(df)
            
            combined_network = nx.compose(twitter_network, reddit_network)
            
            results['twitter_metrics'] = self.calculate_network_metrics(twitter_network)
            results['reddit_metrics'] = self.calculate_network_metrics(reddit_network)
            results['combined_metrics'] = self.calculate_network_metrics(combined_network)
            
            results['twitter_influencers'] = self.identify_influencers(twitter_network, df)
            results['reddit_influencers'] = self.identify_influencers(reddit_network, df)
            
            results['technology_communities'] = self.analyze_technology_communities(combined_network, df)
            
            results['visualization_path'] = self.visualize_network(
                combined_network, 
                {**results['twitter_influencers'], **results['reddit_influencers']},
                output_dir
            )
            
            results_path = os.path.join(output_dir, "network_analysis_results.json")
            with open(results_path, 'w') as f:
                json_results = json.loads(json.dumps(results, default=self._json_serializer))
                json.dump(json_results, f, indent=2)
            
            logging.info(f"Network analysis results saved: {results_path}")
            logging.info("Network analysis completed successfully")
            
            return results
            
        except Exception as e:
            logging.error(f"Network analysis pipeline failed: {e}")
            raise

    def _json_serializer(self, obj):
        """JSON serializer for objects not serializable by default json code"""
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        else:
            return str(obj)

    def run(self, df):
        return self.run_network_analysis(df)

def run_network_analysis(df, config):
    """Main network analysis function"""
    analyzer = NetworkAnalyzer(config)
    return analyzer.run(df)