import pytest
import numpy as np
from ..src.networks import SimplePaymentNetwork, GroupedPaymentNetwork

# Constants for tests
INIT_BANKS = 3
INCREMENT = 4
TOTAL_BANKS = 9
ALPHA = 0.00001
AVG_PAYMENTS = 1000
ALLOW_SELF_LOOP = False
BANK_GROUPS = [10, 35, 23, 42]
TOTAL_BANKS_GROUPED = 25
AVG_PAYMENTS_GROUPED = 100
ALPHA_GROUPED = 0.01
INCREMENT_GROUPED = 25
INIT_BANKS_GROUPED = 5

@pytest.fixture
def simple_network():
    return SimplePaymentNetwork(
        total_banks=TOTAL_BANKS,
        avg_payments=AVG_PAYMENTS,
        alpha=ALPHA,
        allow_self_loop=ALLOW_SELF_LOOP
    )

@pytest.fixture
def grouped_network():
    return GroupedPaymentNetwork(
        total_banks=TOTAL_BANKS_GROUPED,
        bank_groups=BANK_GROUPS,
        avg_payments=AVG_PAYMENTS_GROUPED,
        alpha=ALPHA_GROUPED,
        allow_self_loop=ALLOW_SELF_LOOP
    )

def test_initial_conditions(simple_network):
    assert simple_network.total_banks == TOTAL_BANKS
    assert simple_network.avg_payments == AVG_PAYMENTS
    assert simple_network.alpha == ALPHA
    assert simple_network.allow_self_loop == ALLOW_SELF_LOOP

def test_payment_simulation(simple_network):
    simple_network.simulate_payments(increment=INCREMENT, init_banks=INIT_BANKS)
    assert len(simple_network.G.nodes()) <= TOTAL_BANKS
    assert simple_network.G.number_of_edges() > 0  # Ensure some links are created

def test_extract_link_matrix(simple_network):
    simple_network.simulate_payments(increment=INCREMENT, init_banks=INIT_BANKS)
    matrix = simple_network.extract_link_matrix(False)
    assert np.isclose(matrix.sum(), TOTAL_BANKS * AVG_PAYMENTS)

def test_grouped_network_initialization(grouped_network):
    assert grouped_network.total_banks == TOTAL_BANKS_GROUPED
    assert len(grouped_network.bank_groups) == len(BANK_GROUPS)

def test_grouped_payment_simulation(grouped_network):
    grouped_network.simulate_payments(increment=INCREMENT_GROUPED, init_banks=INIT_BANKS_GROUPED)
    assert len(grouped_network.G.nodes()) <= TOTAL_BANKS_GROUPED
    assert grouped_network.G.number_of_edges() > 0  # Ensure some links are created

def test_grouped_extract_link_matrix(grouped_network):
    grouped_network.simulate_payments(increment=INCREMENT_GROUPED, init_banks=INIT_BANKS_GROUPED)
    matrix = grouped_network.extract_link_matrix()
    assert np.isclose(matrix.sum(), 1.0)  # Check if matrix elements sum to 1 for proportions