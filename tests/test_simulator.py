import pytest
from ..src.simulator import (
    AnomalyGenerator,
    AnomalyTransactionSim,
    SimplePaymentNetwork,
    TransactionSim,
    random_payment_timing,
    random_payment_value,
)

# Constants for tests
SIM_PERIODS = list(range(15))
SIM_PARAMS = {
    "open_time": "06:30:00",
    "close_time": "18:30:00",
    "value_fn": random_payment_value,
    "timing_fn": random_payment_timing,
}
TOTAL_BANKS = 10
AVG_PAYMENTS = 15
ALPHA = 0.01
ALLOW_SELF_LOOP = False
ANOMALY_PARAMS = {
    "anomaly_start": 5,
    "anomaly_end": 10,
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
    return TransactionSim(sim_id=1, network=payment_network, **SIM_PARAMS)


@pytest.fixture
def anomaly_transactions(payment_network):
    anomaly_generator = AnomalyGenerator(**ANOMALY_PARAMS)
    return AnomalyTransactionSim(
        sim_id=2, network=payment_network, anomaly=anomaly_generator, **SIM_PARAMS
    )


def test_normal_transactions(normal_transactions):
    normal_transactions.run(SIM_PERIODS)
    payments_df = normal_transactions.get_payments_df()
    assert not payments_df.empty
    assert payments_df["Value"].sum() > 0


def test_anomaly_transactions(anomaly_transactions):
    anomaly_transactions.run(SIM_PERIODS)
    payments_df = anomaly_transactions.get_payments_df()
    assert not payments_df.empty
    assert payments_df["Value"].sum() > 0
    assert (
        payments_df["Value"].sum()
        > normal_transactions.get_payments_df()["Value"].sum()
    )


def test_success_rate(normal_transactions, anomaly_transactions):
    correct = 0
    test_len = 50
    for _ in range(test_len):
        normal_transactions.run(SIM_PERIODS)
        anomaly_transactions.run(SIM_PERIODS)
        if (
            anomaly_transactions.get_payments_df()["Value"].sum()
            > normal_transactions.get_payments_df()["Value"].sum()
        ):
            correct += 1
    assert (
        correct / test_len > 0.5
    )  # Assuming anomalous transactions should generally be higher


# More specific tests can be added to validate the integrity of individual transactions,
# the distribution of transactions over time, or the handling of edge cases.
