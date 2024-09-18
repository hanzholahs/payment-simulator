import datetime

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from src.payment_simulator.anomaly import AnomalyGenerator
from src.payment_simulator.networks import AbstractPaymentNetwork, SimplePaymentNetwork
from src.payment_simulator.simulator import RTGSNetworkSim
from src.payment_simulator.constants import PAYMENTS_COLNAMES

# Constants for tests
TOTAL_PERIOD = 35
TOTAL_BANKS = 10
TOTAL_PAYMENTS = 100
ALPHA = 0.01
ALLOW_SELF_LOOP = False
OPEN_TIME = "06:30:00"
CLOSE_TIME = "18:30:00"
SIM_PARAMS = {
    "open_time": OPEN_TIME,
    "close_time": CLOSE_TIME,
    "value_fn": np.random.lognormal,
    "time_fn": np.random.uniform,
}
ANOMALY_PARAMS = {
    "anomaly_start": 15,
    "anomaly_end": 30,
    "prob_start": 0.8,
    "prob_end": 1,
    "lambda_start": 0.5,
    "lambda_end": 2.5,
    "rate": 0.5,
}


@pytest.fixture
def payment_network():
    return SimplePaymentNetwork(
        total_banks=TOTAL_BANKS,
        total_payments=TOTAL_PAYMENTS,
        alpha=ALPHA,
        allow_self_loop=ALLOW_SELF_LOOP,
    )


@pytest.fixture
def normal_transactions(payment_network):
    return RTGSNetworkSim(network=payment_network, **SIM_PARAMS)


@pytest.fixture
def anomaly_gen():
    return AnomalyGenerator(**ANOMALY_PARAMS)


def test_initial_conditions(normal_transactions):
    t_open = datetime.datetime.strptime(OPEN_TIME, "%H:%M:%S").time()
    t_close = datetime.datetime.strptime(CLOSE_TIME, "%H:%M:%S").time()

    # check initialization of normal transaction sim
    assert normal_transactions.sim_id == 0
    assert normal_transactions.network is not None
    assert normal_transactions.t_open == t_open
    assert normal_transactions.t_close == t_close
    assert normal_transactions.value_fn == np.random.lognormal
    assert normal_transactions.time_fn == np.random.uniform

    assert isinstance(normal_transactions.network, AbstractPaymentNetwork)
    assert isinstance(normal_transactions.t_open, datetime.time)
    assert isinstance(normal_transactions.t_close, datetime.time)


def test_transactions_df(normal_transactions):
    t_open = datetime.datetime.strptime(OPEN_TIME, "%H:%M:%S").time()
    t_close = datetime.datetime.strptime(CLOSE_TIME, "%H:%M:%S").time()

    # Run normal_transactions and get DataFrame
    normal_transactions.run(TOTAL_PERIOD, daily_payments=TOTAL_PAYMENTS)
    assert isinstance(normal_transactions.network.G, nx.Graph)

    payments_df = normal_transactions.get_payments_df()

    # Validate DataFrame structure
    assert isinstance(payments_df, pd.DataFrame)
    assert not payments_df.empty
    assert (payments_df.columns.values == PAYMENTS_COLNAMES).all()

    # Validate data types in DataFrame columns
    assert np.issubdtype(payments_df["Time"].dtype, datetime.time)
    assert payments_df["Period"].dtype == int
    assert payments_df["Sender"].dtype == int
    assert payments_df["Receiver"].dtype == int
    assert payments_df["Value"].dtype == float

    # Check for non-negative payment counts and values
    assert (payments_df["Value"] > 0).all()

    # Validate operational hour constraints
    assert (payments_df["Time"] >= t_open).all()
    assert (payments_df["Time"] <= t_close).all()

    # Ensure all senders and receivers are valid network nodes
    participants = list(normal_transactions.network.G.nodes())
    assert np.isin(payments_df["Sender"], participants).all()
    assert np.isin(payments_df["Receiver"], participants).all()

    # Confirm the expected number of normal_transactions
    assert payments_df.shape[0] == TOTAL_PAYMENTS * TOTAL_PERIOD


def test_edge_cases(normal_transactions):
    # Test with zero transactions
    normal_transactions.run(TOTAL_PERIOD, daily_payments=0)
    payments_df = normal_transactions.get_payments_df()
    assert payments_df.empty

    # Test with a small number of payments
    normal_transactions.network.total_payments = 1
    normal_transactions.run(TOTAL_PERIOD)
    payments_df = normal_transactions.get_payments_df()
    assert len(payments_df) == 1 * TOTAL_PERIOD


def test_anomaly_rate(payment_network, anomaly_gen):
    sim = RTGSNetworkSim(anomaly_v=anomaly_gen, network=payment_network, **SIM_PARAMS)
    sim.run(TOTAL_PERIOD, anomalous_bank=(1, 2, 3, 4, 5))
    anomaly_df = sim.get_payments_df()
    assert anomaly_df["Anomaly Value"].sum() > 0
    assert (anomaly_df["Anomaly Time"].notna()).sum() > 0


def test_error(normal_transactions):
    with pytest.raises(Exception):
        normal_transactions.get_payments_df()
    with pytest.raises(Exception):
        normal_transactions.get_balances_df()
