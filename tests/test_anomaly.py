import numbers

import pytest

from src.payment_simulator.anomaly import AbstractAnomalyGenerator, AnomalyGenerator

# Constants for tests
TOTAL_PERIOD = 2000
ANOMALY_START = 101
ANOMALY_END = 200
PROB_START = 0.7
PROB_END = 0.9
LAMBDA_START = 100
LAMBDA_END = 1000
RATE = 0.2


@pytest.fixture
def anomaly_generator():
    return AnomalyGenerator(
        anomaly_start=ANOMALY_START,
        anomaly_end=ANOMALY_END,
        prob_start=PROB_START,
        prob_end=PROB_END,
        lambda_start=LAMBDA_START,
        lambda_end=LAMBDA_END,
        rate=LAMBDA_END,
    )


def test_initial_conditions(anomaly_generator):
    # Check if the anomaly generator is correctly instantiated
    assert isinstance(anomaly_generator, AbstractAnomalyGenerator)
    assert isinstance(anomaly_generator, AnomalyGenerator)

    # Ensure all initial settings are correct
    assert anomaly_generator.anomaly_start == ANOMALY_START
    assert anomaly_generator.anomaly_end == ANOMALY_END
    assert anomaly_generator.prob_start == PROB_START
    assert anomaly_generator.prob_end == PROB_END
    assert anomaly_generator.lambda_start == LAMBDA_START
    assert anomaly_generator.lambda_end == LAMBDA_END
    assert (
        anomaly_generator.rate == LAMBDA_END
    )  # Check if 'rate' should be 'LAMBDA_END'


def test_output_type(anomaly_generator):
    # Test that the anomaly generator outputs a real number
    assert isinstance(anomaly_generator(0), numbers.Real)
    assert isinstance(anomaly_generator(0.0), numbers.Real)


def test_run(anomaly_generator):
    # Test the anomaly generator's output over several runs for stochastic behavior
    half_period = int(TOTAL_PERIOD / 2)
    min_anomalies = 50  # Minimum expected anomalies
    max_anomalies = 100  # Maximum expected anomalies

    for _ in range(100):  # Run the generator multiple times
        anomaly_count = 0

        for i in range(-half_period, half_period):
            output = anomaly_generator(i)

            # Assert that outputs outside the anomaly period are zero
            if i < ANOMALY_START or i > ANOMALY_END:
                assert output == 0.0

            # Count non-zero outputs to track anomalies
            if output != 0:
                anomaly_count += 1

        # Check if anomalies are within the expected range
        assert min_anomalies <= anomaly_count <= max_anomalies
