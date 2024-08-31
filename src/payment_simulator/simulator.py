import datetime
from abc import ABC
from typing import Any, Callable, Iterable

import pandas as pd

from .networks import AbstractPaymentNetwork
from .utils import zero_anomaly_gen


class AbstractTransactionSim(ABC):
    """
    An abstract base class designed for simulating transaction scenarios within a financial network.
    This class is intended to be subclassed to provide specific implementations for simulating
    transaction values and timings within operational hours.

    Parameters
    ----------
    network : AbstractPaymentNetwork
        An instance of a payment network which handles the simulation of payments.
    value_fn : Callable
        Function to calculate the value of each transaction.
    timing_fn : Callable
        Function to determine the timing of each transaction within the specified operational hours.
    open_time : str, optional
        The opening time of the transaction window each day, formatted as "HH:MM:SS", default is "08:00:00".
    close_time : str, optional
        The closing time of the transaction window each day, formatted as "HH:MM:SS", default is "17:00:00".

    Attributes
    ----------
    payments : list[tuple]
        A list to store all payment transactions. Each transaction is stored as a tuple containing
        period, timing, sender, receiver, count, and value.
    network : AbstractPaymentNetwork
        The payment network instance used for simulating transactions.
    value_fn : Callable
        Function used to calculate the transaction value.
    timing_fn : Callable
        Function used to determine the transaction timing.
    open_time : datetime.time
        Parsed opening time for transactions.
    close_time : datetime.time
        Parsed closing time for transactions.

    Methods
    -------
    get_payments_df() -> pd.DataFrame
        Returns a DataFrame containing all simulated transactions with details.
    get_balances_df() -> pd.DataFrame
        Returns a DataFrame containing the balances of all participants after simulation.
    simulate_day(init_banks: int | None = None)
        Simulates a day's transactions using the network's payment simulation functionality.
    """

    def __init__(
        self,
        network: AbstractPaymentNetwork,
        value_fn: Callable,
        timing_fn: Callable,
        open_time: str = "08:00:00",
        close_time: str = "17:00:00",
    ) -> None:
        """
        Initializes the transaction simulator with required parameters for managing transaction values
        and timings within operational hours.

        Parameters
        ----------
        network : AbstractPaymentNetwork
            The network instance over which the transactions will be simulated.
        value_fn : Callable
            A function that calculates the monetary value of each transaction.
        timing_fn : Callable
            A function that determines the timing of each transaction within the specified hours.
        open_time : str, optional
            The starting time of the transaction window each day, formatted as "HH:MM:SS". Default is "08:00:00".
        close_time : str, optional
            The ending time of the transaction window each day, formatted as "HH:MM:SS". Default is "17:00:00".
        """
        self.payments: list[tuple] | None = None
        self.balances: list[tuple] | None = None
        self.network = network
        self.value_fn = value_fn
        self.timing_fn = timing_fn
        self.open_time = datetime.datetime.strptime(open_time, "%H:%M:%S").time()
        self.close_time = datetime.datetime.strptime(close_time, "%H:%M:%S").time()

    def get_payments_df(self) -> pd.DataFrame:
        """
        Compiles and returns a DataFrame from the accumulated payment transactions.

        Returns
        -------
        pd.DataFrame
            A DataFrame detailing all transactions, with columns for the period, time of transaction,
            sender bank, receiver bank, count of transactions, and the value of each transaction.

        Raises
        ------
        AssertionError
            If no transactions have been simulated yet, prompting a reminder to run simulations.
        """
        assert self.payments is not None, "`TransactionSim.run()` must be called first."
        col_names = ["Period", "Time", "Sender", "Receiver", "Count", "Value"]
        return pd.DataFrame(self.payments, columns=col_names)

    def get_balances_df(self) -> pd.DataFrame:
        """
        Constructs and returns a DataFrame containing the balances of all participants after simulation.

        Returns
        -------
        pd.DataFrame
            A DataFrame detailing the balances of each participant, with columns for the participant identifier
            and their respective balance.

        Raises
        ------
        AssertionError
            If no simulation has been run yet, indicating that there are no balances to report. This assertion ensures
            that balance information is only attempted to be retrieved after at least one simulation cycle.
        """
        assert self.payments is not None, "`TransactionSim.run()` must be called first."
        col_names = ["Participant", "Balance"]
        return pd.DataFrame(self.balances, columns=col_names)

    def simulate_day(self, init_banks: int | None = None):
        """
        Simulates transaction activities for a single day, optionally initializing a specific number of banks
        at the start of the simulation. This method utilizes the network's simulate_payments method to process transactions.

        Parameters
        ----------
        init_banks : int, optional
            The number of banks to include at the beginning of the day's simulation. If not specified, the simulation
            will proceed with the default setup as defined by the network's initial configuration.

        Notes
        -----
        This method is designed to be run multiple times if simulating transactions over multiple days. Each call
        represents a single day of transaction activity, with the state of the network carrying over to the next call.
        """
        self.network.simulate_payments(init_banks)


class TransactionSim(AbstractTransactionSim):
    """
    A simulation class designed for generating and analyzing anomalous transaction patterns within a financial network.
    This class extends `AbstractTransactionSim` by integrating an anomaly generator to simulate transaction anomalies.

    Parameters
    ----------
    sim_id : Any
        A unique identifier for the simulation, which can be of any data type (int, string, etc.).
    anomaly_gen : Callable
        An instance of an anomaly generator that applies modifications to transaction values, simulating anomalies.
    **kwargs : dict
        Additional keyword arguments that are passed to the superclass `AbstractTransactionSim`, including network, value_fn, timing_fn, open_time, and close_time.

    Attributes
    ----------
    sim_id : Any
        Stores the identifier for this particular simulation instance.
    anomaly_gen : Callable
        Holds the anomaly generator which is used to introduce anomalies into the transactions during the simulation.
    payments : list[tuple]
        Accumulates all transaction data generated during the simulation, capturing details such as period, timing, sender, receiver, count, and anomalous values.

    Methods
    -------
    run(sim_periods: Iterable[int]) -> None
        Executes the simulation over specified time periods, generating payments by integrating anomalies at defined intervals.
    """

    def __init__(
        self, sim_id: Any = 0, anomaly_gen: Callable = zero_anomaly_gen, *args, **kwargs
    ) -> None:
        """
        Initializes the TransactionSim class with a simulation identifier and an anomaly generator, along with other necessary parameters from the base class.

        Parameters
        ----------
        sim_id : Any
            A unique identifier for the simulation.
        anomaly_gen : Callable
            A function or generator instance that will be used to inject anomalies into the transaction values during the simulation.
        *args : tuple
            Positional arguments passed to the superclass.
        **kwargs : dict
            Keyword arguments passed to the superclass, which include network configuration and functions for transaction value and timing.
        """
        super().__init__(*args, **kwargs)
        self.sim_id = sim_id
        self.anomaly_gen = anomaly_gen

    def run(
        self, sim_periods: Iterable[int], anomalous_bank: Iterable[int] = []
    ) -> None:
        """
        Executes the simulation over a series of defined time periods, generating transactions
        and optionally incorporating anomalies for specified banks.

        Parameters
        ----------
        sim_periods : Iterable[int]
            A list of time periods over which the simulation is run, typically representing each day.
        anomalous_bank : Iterable[int], optional
            A list of bank indices that should have anomalies applied to their transactions.
            If empty, no anomalies are applied. Defaults to an empty list.

        Process
        -------
        1. For each period in `sim_periods`, simulate the day's network dynamics.
        2. Iterate over each link (bank pair) in the payment network:
           a. For each link, simulate the specified number of transactions (based on link weight).
           b. Calculate the timing for each transaction within operational hours.
           c. Calculate the transaction value, and apply an anomaly if the sender is listed in `anomalous_bank`.
           d. Store each transaction in `all_payments` with its period, timing, sender, receiver, type, and value.
           e. Update the sender's and receiver's balance accordingly and track the minimum balance for each bank.

        Notes
        -----
        - This method updates the `payments` attribute with all transactions from the simulation,
          where each transaction includes details such as period, timing, sender, receiver, and value.
        - It also updates the `balances` attribute to reflect the minimum balance each bank reaches during the simulation.
        """
        all_payments: list[tuple] = []
        balance_now = [0 for _ in range(self.network.total_banks)]
        balance_min = [0 for _ in range(self.network.total_banks)]

        # Process transactions for each simulation period
        for period in sim_periods:
            self.simulate_day()  # Simulate network dynamics for the day

            # Process each link in the simulated payment network
            for (sender, receiver), data in self.network.G.edges.items():
                # Simulate transactions based on the weight of each link
                for _ in range(data["weight"]):
                    # Calculate transaction timing
                    timing = self.timing_fn(self.open_time, self.close_time)

                    # Calculate transaction value with anomaly
                    value = self.value_fn()
                    if sender in anomalous_bank:
                        value += self.anomaly_gen(period)

                    # Store transaction details
                    all_payments.append((period, timing, sender, receiver, 1, value))

                    # Track balance and min balance
                    balance_now[sender] -= value
                    balance_now[receiver] += value
                    if balance_now[sender] < balance_min[sender]:
                        balance_min[sender] = balance_now[sender]

        self.balances = [(i, -bal) for i, bal in enumerate(balance_min)]
        self.payments = all_payments
