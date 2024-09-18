import datetime
from typing import Any

import networkx as nx
import numpy as np


def anomaly_parameter(
    x_start: float,
    x_end: float,
    rate: float,
    current: int,
    anomaly_start: int,
    anomaly_end: int,
) -> float:
    if current < anomaly_start or anomaly_end < current:
        return 0
    return (
        x_start
        + (x_end - x_start)
        * ((current - anomaly_start) / (anomaly_end - anomaly_start)) ** rate
    )


def to_time(x: float, open: datetime.time, close: datetime.time) -> datetime.time:
    open_datetime = datetime.datetime.combine(datetime.date.today(), open)
    close_datetime = datetime.datetime.combine(datetime.date.today(), close)
    operation_duration = (close_datetime - open_datetime).seconds
    random_period = datetime.timedelta(seconds=int(x * operation_duration))
    return (open_datetime + random_period).time()


def calc_num_payments(G: nx.DiGraph) -> int:
    return np.sum([data["s"] for _, data in G.edges.items()])


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

    return {
        "Number of nodes": num_nodes,
        "Number of links": num_links,
        "Connectivity": connectivity,
        "Reciprocity": reciprocity,
        "Average Degree (k)": avg_degree,
        "Max (k-in)": max_k_in,
        "Max (k-out)": max_k_out,
    }


def zero_fn(period) -> int:
    return 0


def is_positive_int(x: Any, var_name: str):
    if isinstance(x, int) and x > 0:
        return
    raise Exception(f"`{x}` must be a positive integer.")