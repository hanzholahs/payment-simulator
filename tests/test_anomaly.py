import pytest
from ..src.anomaly import TransactionSimulator, AnomalyGenerator
import numpy as np

# Constants for tests
SIM_PERIODS = 30
TOTAL_BANKS = 20
AVG_PAYMENTS = 100
ALPHA = 0.01
ALLOW_SELF_LOOP = True

@pytest.fixture
def transaction_simulator():
    return TransactionSimulator(
        total_banks=TOTAL_BANKS,
        avg_payments=AVG_PAYMENTS,
        alpha=ALPHA,
        allow_self_loop=ALLOW_SELF_LOOP
    )

@pytest.fixture
def anomaly_generator():
    return AnomalyGenerator(
        anomaly_start=10,
        anomaly_end=20,
        prob_start=0.1,
        prob_end=0.9,
        lambda_start=1,
        lambda_end=3,
        rate=0.5
    )

def test_initial_conditions(transaction_simulator, anomaly_generator):
    assert transaction_simulator.total_banks == TOTAL_BANKS
    assert anomaly_generator.anomaly_start == 10
    assert anomaly_generator.prob_end == 0.9

def test_transaction_simulation(transaction_simulator):
    # Simulate transactions and check for basic correctness
    transaction_simulator.simulate_transactions(SIM_PERIODS)
    transactions = transaction_simulator.get_transactions()
    assert len(transactions) == SIM_PERIODS * TOTAL_BANKS * AVG_PAYMENTS
    assert all(t >= 0 for t in transactions)  # Assuming transactions values are non-negative

def test_anomaly_injection(transaction_simulator, anomaly_generator):
    # Simulate transactions with anomalies
    transaction_simulator.simulate_transactions(SIM_PERIODS, anomaly_generator)
    transactions = transaction_simulator.get_transactions()
    normal_transactions = transaction_simulator.get_transactions()  # Assuming a way to get transactions without anomalies for comparison
    assert np.mean(transactions) > np.mean(normal_transactions)  # Check if anomalies increased the transaction values on average

def test_output_integrity(transaction_simulator):
    transaction_simulator.simulate_transactions(SIM_PERIODS)
    transactions = transaction_simulator.get_transactions()
    assert isinstance(transactions, list)  # Assuming transactions are stored as a list
    assert all(isinstance(t, float) for t in transactions)  # Assuming transactions are floating-point numbers

def test_edge_cases(transaction_simulator):
    # Test with zero transactions
    transaction_simulator.avg_payments = 0
    transaction_simulator.simulate_transactions(SIM_PERIODS)
    transactions = transaction_simulator.get_transactions()
    assert len(transactions) == 0

    # Test with the maximum number of banks
    transaction_simulator.total_banks = 1000  # Arbitrary large number
    transaction_simulator.avg_payments = 1  # Reducing payments to manage performance
    transaction_simulator.simulate_transactions(1)  # Only one period for performance
    transactions = transaction_simulator.get_transactions()
    assert len(transactions) == 1000
