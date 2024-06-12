import datetime
import numpy as np
import networkx as nx
def calculate_network_params(G: nx.DiGraph) -> dict:
    """
    Calculates and returns various parameters of the simulation such as connectivity, reciprocity, and degrees.

    :return: Dictionary containing calculated simulation parameters.
    """        
    num_nodes = G.number_of_nodes()
    num_links = G.number_of_edges()
    connectivity = num_links / (num_nodes * (num_nodes - 1))
    reciprocity = nx.reciprocity(G)
    
    avg_degree = np.mean([val for _, val in G.degree])
    max_k_in = np.max([val for _, val in G.in_degree])
    max_k_out = np.max([val for _, val in G.out_degree])

    num_payments = np.sum([data['s'] for _, data in G.edges.items()])
    
    return {
        "Number of nodes": num_nodes,
        "Number of links": num_links,
        "Connectivity": connectivity,
        "Reciprocity": reciprocity,
        "Average Degree (k)": avg_degree,
        "Max (k-in)": max_k_in,
        "Max (k-out)": max_k_out,
        "Number of payments": num_payments
    }
