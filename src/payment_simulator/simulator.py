import abc
import datetime
from itertools import repeat
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd

from .constants import BALANCES_COLNAMES, PAYMENTS_COLNAMES
from .networks import AbstractPaymentNetwork, SimplePaymentNetwork
from .utils import to_time, zero_fn


class AbstractRTGSTransactions(abc.ABC):
    """An abstract base class designed for simulating transaction scenarios within a financial network.

    This class is intended to be subclassed to provide specific implementations for simulating
    transaction values and times within operational hours.

    Attributes
    ----------
    payments : Optional[List[Tuple]]
        Stores the details of all simulated transactions.
    balances : Optional[List[Tuple]]
        Stores the balances of all participants after the simulation.

    Methods
    -------
    run()
        Executes the simulation process.
    get_payments_df() -> pd.DataFrame
        Returns a DataFrame containing all simulated transactions with details.
    get_balances_df() -> pd.DataFrame
        Returns a DataFrame containing the balances of all participants after simulation.
    """

    def __init__(self) -> None:
        """Initializes the transaction simulator with required parameters for managing transaction values
        and times within operational hours.
        """
        self.payments: Optional[list[tuple]] = None
        self.balances: Optional[list[tuple]] = None

    @abc.abstractmethod
    def run(self) -> None:
        """Executes the simulation process.

        This abstract method should be implemented by subclasses to define how the simulation runs.
        """
        pass

    @abc.abstractmethod
    def get_payments_df(self) -> pd.DataFrame:
        """Generates and returns a DataFrame containing all simulated transactions with details.

        Returns
        -------
        pd.DataFrame
            A DataFrame with details of all simulated transactions, including timestamps,
            amounts, sender, and receiver information.
        """
        pass

    @abc.abstractmethod
    def get_balances_df(self) -> pd.DataFrame:
        """Generates and returns a DataFrame containing the balances of all participants after simulation.

        Returns
        -------
        pd.DataFrame
            A DataFrame with the final balances of each participant in the network after the simulation
            has been run.
        """
        pass


class RTGSNetworkSim(AbstractRTGSTransactions):
    """A simulation class designed for generating and analyzing anomalous transaction patterns within a financial network.
    
    This class extends `AbstractRTGSTransactions` by integrating anomaly generators to simulate transaction anomalies.

    Attributes
    ----------
    sim_id : int or str
        An identifier for the simulation instance.
    network : AbstractPaymentNetwork
        The payment network over which transactions are simulated.
    time_fn : Callable
        A function that generates transaction times within operational hours.
    value_fn : Callable
        A function that generates transaction values.
    anomaly_v : Callable
        A function that generates anomalous transaction values.
    anomaly_t : Callable
        A function that generates anomalous transaction times.
    t_open : datetime.time
        The opening time of the operational hours.
    t_close : datetime.time
        The closing time of the operational hours.
    payments : Optional[List[Tuple]]
        Stores the details of all simulated transactions.
    balances : Optional[List[Tuple]]
        Stores the balances of all participants after the simulation.

    Methods
    -------
    run(total_period=1, anomalous_bank=(), daily_payments=None, **sim_params)
        Executes the simulation over specified time periods, generating payments and integrating anomalies at defined intervals.
    get_payments_df() -> pd.DataFrame
        Compiles and returns a DataFrame from the accumulated payment transactions.
    get_balances_df() -> pd.DataFrame
        Constructs and returns a DataFrame containing the balances of all participants after simulation.
    """

    def __init__(
        self,
        sim_id: int | str = 0,
        network: AbstractPaymentNetwork = SimplePaymentNetwork(10, 2),
        time_fn: Callable = np.random.uniform,
        value_fn: Callable = np.random.lognormal,
        anomaly_v: Callable = zero_fn,
        anomaly_t: Callable = zero_fn,
        open_time: str = "08:00:00",
        close_time: str = "17:00:00",
    ) -> None:
        """Initializes the RTGSNetworkSim class with a simulation identifier, payment network, anomaly generators, and operational hours.

        Parameters
        ----------
        sim_id : int or str, optional
            An identifier for the simulation instance, by default 0.
        network : AbstractPaymentNetwork, optional
            The payment network over which transactions are simulated, by default `SimplePaymentNetwork(10, 2)`.
        time_fn : Callable, optional
            A function to generate transaction times within operational hours, by default `np.random.uniform`.
        value_fn : Callable, optional
            A function to generate transaction values, by default `np.random.lognormal`.
        anomaly_v : Callable, optional
            A function to generate anomalous transaction values, by default `zero_fn`.
        anomaly_t : Callable, optional
            A function to generate anomalous transaction times, by default `zero_fn`.
        open_time : str, optional
            The opening time of the operational hours in "HH:MM:SS" format, by default "08:00:00".
        close_time : str, optional
            The closing time of the operational hours in "HH:MM:SS" format, by default "17:00:00".
        """
        super().__init__()
        self.sim_id = sim_id
        self.network = network
        self.time_fn = time_fn
        self.value_fn = value_fn
        self.anomaly_v = anomaly_v
        self.anomaly_t = anomaly_t
        self.t_open = datetime.datetime.strptime(open_time, "%H:%M:%S").time()
        self.t_close = datetime.datetime.strptime(close_time, "%H:%M:%S").time()

    def run(
        self,
        total_period: int = 1,
        anomalous_bank: Iterable[int] = (),
        daily_payments: Iterable[int] | int | None = None,
        **sim_params,
    ) -> None:
        """Executes the simulation over a series of defined time periods, generating transactions
        and optionally incorporating anomalies for specified banks.

        Parameters
        ----------
        total_period : int, optional
            The total number of periods (e.g., days) over which to run the simulation, by default 1.
        anomalous_bank : Iterable[int], optional
            An iterable of bank identifiers that will have anomalous transactions, by default `()`.
        daily_payments : Iterable[int] or int or None, optional
            Specifies the number of payments per period. If an iterable is provided, it should have a length equal to `total_period`.
            If an int is provided, it is used for all periods. If `None`, defaults to the network's total payments, by default `None`.
        **sim_params
            Additional parameters to pass to the network's payment simulation method.

        Process
        -------
        1. For each period in `total_period`, simulate the day's network dynamics.
        2. Iterate over each link (bank pair) in the payment network:
           a. For each link, simulate the specified number of transactions (based on link weight).
           b. Calculate the time for each transaction within operational hours.
           c. Calculate the transaction value, and apply an anomaly if the sender is listed in `anomalous_bank`.
           d. Store each transaction in `all_payments` with its period, time, sender, receiver, type, and value.
           e. Update the sender's and receiver's balance accordingly and track the minimum balance for each bank.

        Notes
        -----
        - This method updates the `payments` attribute with all transactions from the simulation,
          where each transaction includes details such as period, time, sender, receiver, and value.
        - It also updates the `balances` attribute to reflect the minimum balance each bank reaches during the simulation.
        """
        if daily_payments is None:
            daily_payments = repeat(self.network.total_payments, total_period)
        elif isinstance(daily_payments, int):
            daily_payments = repeat(daily_payments, total_period)

        all_payments: list[tuple] = []
        balance_now = [0 for _ in range(self.network.total_banks)]
        balance_min = [0 for _ in range(self.network.total_banks)]

        # Process transactions for each simulation period
        for period, total_payments in zip(range(total_period), daily_payments):
            # Simulate network dynamics for the day
            self.network.total_payments = total_payments
            self.network.simulate_payments(**sim_params)

            # Process each link in the simulated payment network
            for (sender, receiver), data in self.network.G.edges.items():
                # Simulate transactions based on the weight of each link
                for _ in range(data["weight"]):
                    is_anomaly = sender in anomalous_bank

                    # Calculate transaction time and value
                    payment_t = to_time(self.time_fn(), self.t_open, self.t_close)
                    payment_v = self.value_fn()

                    # Calculate transaction time and value anomaly
                    anomaly_t = to_time(
                        self.anomaly_t(period), self.t_open, self.t_close
                    )
                    anomaly_v = self.anomaly_v(period)

                    # Store transaction details
                    all_payments.append(
                        (
                            period,
                            sender,
                            receiver,
                            payment_t,
                            payment_v,
                            anomaly_t if is_anomaly else None,
                            is_anomaly * anomaly_v,
                        )
                    )

                    # Track sender and receiver balance
                    balance_now[sender] -= payment_v
                    balance_now[receiver] += payment_v

                    # Update sender minimal balance
                    if balance_now[sender] < balance_min[sender]:
                        balance_min[sender] = balance_now[sender]

        self.balances = [(i, -bal) for i, bal in enumerate(balance_min)]
        self.payments = all_payments

    def get_payments_df(self) -> pd.DataFrame:
        """Compiles and returns a DataFrame from the accumulated payment transactions.

        Returns
        -------
        pd.DataFrame
            A DataFrame detailing all transactions, with columns for the period, sender bank, receiver bank, transaction time,
            transaction value, anomaly time (if any), and anomaly value (if any).

        Raises
        ------
        Exception
            If no transactions have been simulated yet, prompting a reminder to run simulations.
        """
        if self.payments is None:
            raise Exception("`TransactionSim.run()` must be called first.")

        return pd.DataFrame(self.payments, columns=PAYMENTS_COLNAMES)

    def get_balances_df(self) -> pd.DataFrame:
        """Constructs and returns a DataFrame containing the balances of all participants after simulation.

        Returns
        -------
        pd.DataFrame
            A DataFrame detailing the balances of each participant, with columns for the participant identifier
            and their respective balance.

        Raises
        ------
        Exception
            If no simulation has been run yet, indicating that there are no balances to report.
        """
        if self.balances is None:
            raise Exception("`TransactionSim.run()` must be called first.")

        return pd.DataFrame(self.balances, columns=BALANCES_COLNAMES)
