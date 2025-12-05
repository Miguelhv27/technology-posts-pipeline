import os
import logging
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import json

class NetworkAnalyzer:
    """
    DataOps Network Analysis Engine.
    Analyzes structure, influence, and communities based on interaction data.
    """

    def __init__(self, config):
        self.config = config
        self.network_config = config["network_analysis"]

    def build_network(self, df):
        """
        Constructs a graph based on available data signatures.
        - Reddit: Bipartite Graph (User <-> Subreddit)
        - Twitter: Simulated Interaction Graph (due to API limitations on follower lists)
        """
        G = nx.Graph()
        
        if df.empty:
            return G

        reddit_df = df[df['data_source'] == 'reddit']
        if not reddit_df.empty:
            logging.info(f"Building Reddit Network from {len(reddit_df)} posts...")
            for _, row in reddit_df.iterrows():
                user = row.get('author', 'unknown')
                sub = row.get('subreddit', 'unknown')
                score = row.get('score', 1)
                
                G.add_node(user, type='user', source='reddit', 
                           sentiment=row.get('sentiment_polarity', 0))
                G.add_node(sub, type='community', source='reddit')
                
                G.add_edge(user, sub, weight=score)

        twitter_df = df[df['data_source'] == 'twitter']
        if not twitter_df.empty:
            logging.info(f"Building Twitter Network (Simulated) from {len(twitter_df)} tweets...")
            users = twitter_df['user_id'].unique()
            
            for _, row in twitter_df.iterrows():
                G.add_node(row['user_id'], type='user', source='twitter',
                           sentiment=row.get('sentiment_polarity', 0),
                           followers=row.get('user_followers_count', 0))
            
            for user in users:
                targets = np.random.choice(users, size=np.random.randint(2, 5))
                for target in targets:
                    if user != target:
                        G.add_edge(user, target, weight=1)

        logging.info(f"Network Built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G

    def identify_influencers(self, G):
        """
        Identifies key nodes using PageRank (Global influence) and Degree (Local influence).
        """
        if G.number_of_nodes() < 5:
            return {}

        try:
            pagerank = nx.pagerank(G, alpha=0.85)
            
            top_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
            
            influencers = {}
            for node, score in top_nodes:
                node_data = G.nodes[node]
                if node_data.get('type') == 'user':
                    influencers[node] = {
                        'rank_score': score,
                        'source': node_data.get('source', 'unknown'),
                        'followers': node_data.get('followers', 0)
                    }
            return influencers

        except Exception as e:
            logging.warning(f"Influencer detection failed: {e}")
            return {}

    def visualize(self, G, output_dir):
        """Creates a simple visualization of the network structure"""
        if G.number_of_nodes() == 0: return None
        
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G, k=0.15, seed=42)
        
        colors = []
        sizes = []
        for node in G.nodes():
            if G.nodes[node].get('type') == 'community':
                colors.append('red') 
                sizes.append(300)
            else:
                colors.append('skyblue') 
                sizes.append(50)

        nx.draw(G, pos, node_color=colors, node_size=sizes, alpha=0.6, with_labels=False)
        
        output_path = os.path.join(output_dir, "network_graph.png")
        plt.title("Interaction Network (Blue=Users, Red=Communities)")
        plt.savefig(output_path)
        plt.close()
        
        return output_path

    def run(self, df, output_dir="outputs/network"):
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.network_config.get("enabled", False):
            return {}

        try:
            G = self.build_network(df)
            
            metrics = {
                "nodes": G.number_of_nodes(),
                "edges": G.number_of_edges(),
                "density": nx.density(G)
            }
            
            influencers = self.identify_influencers(G)
            
            viz_path = self.visualize(G, output_dir)
            
            results = {
                "metrics": metrics,
                "top_influencers": influencers,
                "visualization": viz_path
            }
            
            with open(os.path.join(output_dir, "network_results.json"), 'w') as f:
                json.dump(results, f, indent=4)
                
            return results

        except Exception as e:
            logging.error(f"Network Analysis failed: {e}")
            return {}

def run_network_analysis(df, config):
    analyzer = NetworkAnalyzer(config)
    return analyzer.run(df)