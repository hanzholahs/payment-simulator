import numpy as np
import networkx as nx
import numbers
import pytest

from src.payment_simulator.networks import (
    AbstractPaymentNetwork,
    GroupedPaymentNetwork,
    SimplePaymentNetwork,
)

# Constants for tests
INIT_BANKS = 3
INCREMENT = 5
TOTAL_BANKS = 35
ALPHA = 0.00001
AVG_PAYMENTS = 100
ALLOW_SELF_LOOP = False
BANK_GROUPS = [10, 35, 23, 42]


@pytest.fixture
def simple_network():
    return SimplePaymentNetwork(
        total_banks=TOTAL_BANKS,
        avg_payments=AVG_PAYMENTS,
        alpha=ALPHA,
        allow_self_loop=ALLOW_SELF_LOOP,
    )


@pytest.fixture
def grouped_network():
    return GroupedPaymentNetwork(
        total_banks=TOTAL_BANKS,
        bank_groups=BANK_GROUPS,
        avg_payments=AVG_PAYMENTS,
        alpha=ALPHA,
        allow_self_loop=ALLOW_SELF_LOOP,
    )


def test_initial_conditions(grouped_network, simple_network):
    # Test SimplePaymentNetwork
    assert isinstance(simple_network, AbstractPaymentNetwork)
    assert isinstance(simple_network, SimplePaymentNetwork)
    assert simple_network.total_banks == TOTAL_BANKS
    assert simple_network.avg_payments == AVG_PAYMENTS
    assert simple_network.alpha == ALPHA
    assert simple_network.allow_self_loop == ALLOW_SELF_LOOP

    # Test GroupedPaymentNetwork
    assert isinstance(grouped_network, AbstractPaymentNetwork)
    assert isinstance(grouped_network, GroupedPaymentNetwork)
    assert grouped_network.total_banks == TOTAL_BANKS
    assert grouped_network.bank_groups == BANK_GROUPS
    assert grouped_network.avg_payments == AVG_PAYMENTS
    assert grouped_network.alpha == ALPHA
    assert grouped_network.allow_self_loop == ALLOW_SELF_LOOP


@pytest.mark.parametrize("network", ["simple_network", "grouped_network"])
def test_payment_simulation(network, request):
    network = request.getfixturevalue(network)
    network.simulate_payments(increment=INCREMENT, init_banks=INIT_BANKS)
    assert isinstance(network.G, nx.DiGraph)
    assert isinstance(network.h, np.ndarray)

    # All banks are simulated in the network
    assert len(network.G.nodes()) == TOTAL_BANKS

    # Preference vector must be at least 1
    assert np.all(network.h >= 1)

    # Ensure some links/transactions are created
    assert network.G.number_of_edges() > 0

    # Check created links/transactions
    for i in range(TOTAL_BANKS):
        # Ensure no self loop
        assert i not in network.G[i]

        # Ensure weight is a number greater than zero
        for j in network.G[i]:
            weight = network.G[i][j]["weight"]
            assert isinstance(weight, numbers.Real)
            assert weight >= 0


@pytest.mark.parametrize("network", ["simple_network", "grouped_network"])
def test_self_loop(network, request):
    network = request.getfixturevalue(network)
    network.allow_self_loop = True
    network.simulate_payments(increment=INCREMENT, init_banks=INIT_BANKS)

    # Check self links/transactions
    self_loops = sum(1 for i in range(TOTAL_BANKS) if i in network.G[i])
    
    # ensure some self loop occurs
    assert self_loops > 0

    # Revert to initial condition (False)
    network.allow_self_loop = ALLOW_SELF_LOOP


@pytest.mark.parametrize("network", ["simple_network", "grouped_network"])
def test_extract_link_matrix(network, request):
    network = request.getfixturevalue(network)
    network.simulate_payments(increment=INCREMENT, init_banks=INIT_BANKS)

    # Ensure link matrix all positive and sums up to the expected total
    matrix = network.extract_link_matrix(False)
    assert np.all(matrix >= 0)
    assert matrix.sum() == TOTAL_BANKS * AVG_PAYMENTS

    # Ensure link matrix probabilities all positive and sum to 1
    matrix_prob = network.extract_link_matrix()
    assert np.all(matrix_prob >= 0)
    assert np.isclose(matrix_prob.sum(), 1, atol=1e-3)