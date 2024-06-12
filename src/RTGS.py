from networks import AbstractPaymentNetwork, SimplePaymentNetwork
from utils import *

import datetime
import numpy as np
import pandas as pd
from abc import ABC



class AbstractRTGSSimulator(ABC):
    def __init__(self) -> None:
        self.network: AbstractPaymentNetwork = None
        self.payments: pd.DataFrame = None

    def get_payments_df(self) -> pd.DataFrame:
        col_names = ["Period", "Sender", "Receiver", "Count", "Value"]
        return pd.DataFrame(self.payments, columns=col_names)

    def simulate_day(self, init_banks: int = None):
        self.network.simulate_payments(init_banks)


class RTGSSimulator(AbstractRTGSSimulator):
    def __init__(self,
                 network,
                 open_time: str = "08:00:00",
                 close_time: str = "17:00:00") -> None:
        super().__init__()
        self.network = network
        self.open_time = datetime.datetime.strptime(open_time, '%H:%M:%S').time()
        self.close_time = datetime.datetime.strptime(close_time, '%H:%M:%S').time() 


    def run(self, sim_dates: list[datetime.datetime]) -> None:
        all_payments = []
        for date in sim_dates:
            self.simulate_day()
            day_start_time = datetime.datetime.combine(date, self.open_time)
            day_close_time = datetime.datetime.combine(date, self.close_time)
            for (i, j), data in self.network.G.edges.items():
                for _ in range(data['s']):
                    period = random_payment_period(day_start_time, day_close_time)
                    value = random_payment_value()
                    all_payments.append((period, i, j, 1, value))
        self.payments = all_payments



if __name__ == "__main__":
    sim_dates = [
        datetime.datetime(2012, 10, 12),
        datetime.datetime(2012, 10, 13),
        datetime.datetime(2012, 10, 14),
    ]

    sim_params = {
        "open_time": "06:30:00",
        "close_time": "18:30:00"
    }

    payment_network = SimplePaymentNetwork(total_banks=10,
                                           avg_payments=1000,
                                           alpha=0.01,
                                           allow_self_loop=False)
    
    simulator = RTGSSimulator(payment_network, **sim_params)
    simulator.run(sim_dates)

    payments = simulator.get_payments_df()
    print(payments)