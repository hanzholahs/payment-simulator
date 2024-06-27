from utils import anomaly_parameter
from abc import ABC, abstractmethod

import numpy as np


class AbstractAnomalyGenerator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, current_period: int) -> float:
        pass


class AnomalyGenerator(AbstractAnomalyGenerator):
    def __init__(
        self,
        anomaly_start: int,
        anomaly_end: int,
        prob_start: float,
        prob_end: float,
        lambda_start: float,
        lambda_end: float,
        rate: float,
    ) -> None:
        self.anomaly_start = anomaly_start
        self.anomaly_end = anomaly_end
        self.prob_start = prob_start
        self.prob_end = prob_end
        self.lambda_start = lambda_start
        self.lambda_end = lambda_end
        self.rate = rate

    def __call__(self, current_period: int) -> float:
        prob_val = anomaly_parameter(
            self.prob_start,
            self.prob_end,
            self.rate,
            current_period,
            self.anomaly_start,
            self.anomaly_end,
        )

        lambda_val = anomaly_parameter(
            self.lambda_start,
            self.lambda_end,
            self.rate,
            current_period,
            self.anomaly_start,
            self.anomaly_end,
        )

        anomaly_prob = np.random.binomial(1, prob_val)
        anomaly_val = np.random.exponential(lambda_val)

        return anomaly_prob * anomaly_val


if __name__ == "__main__":
    gen = AnomalyGenerator(
        anomaly_start=10,
        anomaly_end=20,
        prob_start=0.2,
        prob_end=0.8,
        lambda_start=1e2,
        lambda_end=1e3,
        rate=0.5,
    )

    for i in range(25):
        print(f"{i:2d} : {gen(i)}")
