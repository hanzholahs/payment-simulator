from networks import AbstractPaymentNetwork, SimplePaymentNetwork
from anomaly import AbstractAnomalyGenerator, AnomalyGenerator
from utils import *

import datetime
import numpy as np
import pandas as pd
from abc import ABC



class AbstractRTGSSimulator(ABC):
    def __init__(self,
                 network: AbstractPaymentNetwork = None,
                 open_time: str = "08:00:00",
                 close_time: str = "17:00:00") -> None:
        self.payments = None
        self.network = network
        self.open_time = datetime.datetime.strptime(open_time, '%H:%M:%S').time()
        self.close_time = datetime.datetime.strptime(close_time, '%H:%M:%S').time() 

    def get_payments_df(self) -> pd.DataFrame:
        col_names = ["Period", "Time", "Sender", "Receiver", "Count", "Value"]
        return pd.DataFrame(self.payments, columns=col_names)

    def simulate_day(self, init_banks: int = None):
        self.network.simulate_payments(init_banks)


class RTGSSimulator(AbstractRTGSSimulator):
    def __init__(self,
                 sim_id,
                 open_time: str = "08:00:00",
                 close_time: str = "17:00:00",
                 network: AbstractPaymentNetwork = None) -> None:
        super().__init__(network, open_time, close_time)
        self.sim_id = sim_id

    def run(self, sim_periods: list[datetime.datetime]) -> None:
        all_payments = []
        for period in sim_periods:
            self.simulate_day()
            for (i, j), data in self.network.G.edges.items():
                for _ in range(data['s']):
                    timing = random_payment_period(self.open_time, self.close_time)
                    value = random_payment_value()
                    all_payments.append((period, timing, i, j, 1, value))
        self.payments = all_payments


class AnomalyRTGSSimulator(AbstractRTGSSimulator):
    def __init__(self,
                 sim_id,
                 open_time: str = "08:00:00",
                 close_time: str = "17:00:00",
                 network: AbstractPaymentNetwork = None,
                 anomaly: AbstractAnomalyGenerator = None) -> None:
        super().__init__(network, open_time, close_time)
        self.sim_id = sim_id
        self.anomaly = anomaly

    def run(self, sim_periods: list[datetime.datetime]) -> None:
        all_payments = []
        for period in sim_periods:
            self.simulate_day()
            for (i, j), data in self.network.G.edges.items():
                for _ in range(data['s']):
                    timing = random_payment_period(self.open_time, self.close_time)
                    value = random_payment_value() + self.anomaly.anomaly_value(period)
                    all_payments.append((period, timing, i, j, 1, value))
        self.payments = all_payments




if __name__ == "__main__":
    sim_periods = list(range(15))

    sim_params = {
        "open_time": "06:30:00",
        "close_time": "18:30:00"
    }

    payment_network = SimplePaymentNetwork(total_banks=10,
                                           avg_payments=15,
                                           alpha=0.01,
                                           allow_self_loop=False)
    
    anomaly_generator = AnomalyGenerator(anomaly_start=5,
                                         anomaly_end=10,
                                         prob_start=0.8,
                                         prob_end=1,
                                         lambda_start=0.5,
                                         lambda_end=2.5,
                                         rate=0.5)
    
    nrml_simulator = RTGSSimulator(sim_id=1, network=payment_network, **sim_params)
    nrml_simulator.run(sim_periods)

    payments1 = nrml_simulator.get_payments_df()
    print(payments1.head(3))
    print(payments1.tail(3))

    anml_simulator = AnomalyRTGSSimulator(sim_id=2, 
                                          network=payment_network,
                                          anomaly=anomaly_generator,
                                          **sim_params)
    anml_simulator.run(sim_periods)

    payments2 = anml_simulator.get_payments_df()
    print(payments2.head(3))
    print(payments2.tail(3))

    print(f"Total Transaction of Normal RTGS  : {payments1['Value'].sum():.3f}")
    print(f"Total Transaction of Anomaly RTGS : {payments2['Value'].sum():.3f}")

    correct = 0
    test_len = 50

    for _ in range(test_len):
        nrml_simulator.run(sim_periods)
        anml_simulator.run(sim_periods)
        x1 = nrml_simulator.get_payments_df()["Value"].sum()
        x2 = anml_simulator.get_payments_df()["Value"].sum()
        if x2 > x1:
            correct += 1

    print(f"Success rate: {correct / test_len * 100:.2f}%")