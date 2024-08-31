import datetime

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from src.payment_simulator.anomaly import AnomalyGenerator
from src.payment_simulator.networks import AbstractPaymentNetwork, SimplePaymentNetwork
from src.payment_simulator.simulator import TransactionSim
from src.payment_simulator.utils import random_payment_timing, random_payment_value

# Constants for tests
TOTAL_BANKS = 10
AVG_PAYMENTS = 15
ALPHA = 0.01
ALLOW_SELF_LOOP = False
SIM_PERIODS = list(range(35))
OPEN_TIME = "06:30:00"
CLOSE_TIME = "18:30:00"
SIM_PARAMS = {
    "open_time": OPEN_TIME,
    "close_time": CLOSE_TIME,
    "value_fn": random_payment_value,
    "timing_fn": random_payment_timing,
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
        avg_payments=AVG_PAYMENTS,
        alpha=ALPHA,
        allow_self_loop=ALLOW_SELF_LOOP,
    )


@pytest.fixture
def normal_transactions(payment_network):
    return TransactionSim(network=payment_network, **SIM_PARAMS)


@pytest.fixture
def anomaly_gen():
    return AnomalyGenerator(**ANOMALY_PARAMS)


def test_initial_conditions(normal_transactions):
    open_time = datetime.datetime.strptime(OPEN_TIME, "%H:%M:%S").time()
    close_time = datetime.datetime.strptime(CLOSE_TIME, "%H:%M:%S").time()

    # check initialization of normal transaction sim
    assert normal_transactions.sim_id == 0
    assert normal_transactions.network is not None
    assert normal_transactions.open_time == open_time
    assert normal_transactions.close_time == close_time
    assert normal_transactions.value_fn == random_payment_value
    assert normal_transactions.timing_fn == random_payment_timing

    assert isinstance(normal_transactions.network, AbstractPaymentNetwork)
    assert isinstance(normal_transactions.open_time, datetime.time)
    assert isinstance(normal_transactions.close_time, datetime.time)


def test_transactions_df(normal_transactions):
    colnames = ["Period", "Time", "Sender", "Receiver", "Count", "Value"]
    open_time = datetime.datetime.strptime(OPEN_TIME, "%H:%M:%S").time()
    close_time = datetime.datetime.strptime(CLOSE_TIME, "%H:%M:%S").time()

    # Run normal_transactions and get DataFrame
    normal_transactions.run(SIM_PERIODS)
    assert isinstance(normal_transactions.network.G, nx.Graph)

    payments_df = normal_transactions.get_payments_df()

    # Validate DataFrame structure
    assert isinstance(payments_df, pd.DataFrame)
    assert not payments_df.empty
    assert (payments_df.columns.values == colnames).all()

    # Validate data types in DataFrame columns
    assert np.issubdtype(payments_df["Time"].dtype, datetime.time)
    assert payments_df["Sender"].dtype == int
    assert payments_df["Receiver"].dtype == int
    assert payments_df["Count"].dtype == int
    assert payments_df["Value"].dtype == float

    # Check for non-negative payment counts and values
    assert (payments_df["Count"] > 0).all()
    assert (payments_df["Value"] > 0).all()

    # Validate operational hour constraints
    assert (payments_df["Time"] >= open_time).all()
    assert (payments_df["Time"] <= close_time).all()

    # Ensure all senders and receivers are valid network nodes
    participants = list(normal_transactions.network.G.nodes())
    assert np.isin(payments_df["Sender"], participants).all()
    assert np.isin(payments_df["Receiver"], participants).all()

    # Confirm the expected number of normal_transactions
    total_normal_transactions = TOTAL_BANKS * AVG_PAYMENTS * len(SIM_PERIODS)
    assert payments_df.shape[0] == total_normal_transactions


def test_edge_cases(normal_transactions):
    # Test with zero transactions
    normal_transactions.network.avg_payments = 0
    normal_transactions.run(SIM_PERIODS)
    payments_df = normal_transactions.get_payments_df()
    assert payments_df.empty

    # Test with the maximum number of banks
    normal_transactions.network.total_banks = 1000  # Arbitrary large number
    normal_transactions.network.avg_payments = (
        1  # Reduce payments to manage performance
    )
    normal_transactions.run(SIM_PERIODS)  # Run only one period for performance reasons
    payments_df = normal_transactions.get_payments_df()
    assert len(payments_df) == 1000 * len(SIM_PERIODS)


def test_anomaly_rate(normal_transactions, payment_network, anomaly_gen):
    correct = 0
    test_len = 20

    anomaly_transactions = TransactionSim(
        sim_id=1, anomaly_gen=anomaly_gen, network=payment_network, **SIM_PARAMS
    )

    for _ in range(5):
        for _ in range(test_len):
            normal_transactions.run(SIM_PERIODS)
            anomaly_transactions.run(SIM_PERIODS, range(3))

            df_normal = normal_transactions.get_payments_df()
            df_anomaly = anomaly_transactions.get_payments_df()

            if df_normal["Value"].sum() < df_anomaly["Value"].sum():
                correct += 1

        # Assuming anomalous transactions should generally be higher
        assert correct / test_len > 0.9


def test_error(normal_transactions):
    with pytest.raises(Exception):
        normal_transactions.get_payments_df()
    with pytest.raises(Exception):
        normal_transactions.get_balances_df()