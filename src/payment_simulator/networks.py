import abc

import networkx as nx
import numpy as np
from numpy.random import randint

from .utils import is_positive_int


class AbstractPaymentNetwork(abc.ABC):
    """
    An abstract base class for creating payment networks that simulate transactions
    between banks within a financial system.

    Attributes
    ----------
    total_banks : int
        The total number of banks included in the network simulation.
    total_payments : int
        The total number of payments to be simulated across the network.
    G : networkx.Graph
        The graph representing the payment network, where nodes represent banks
        and edges represent transactions between them.
    
    Methods
    -------
    simulate_payments()
        Abstract method to simulate payments between banks in the network.
        Must be implemented by subclasses.
    extract_link_matrix(prop: bool = True) -> np.ndarray
        Returns the adjacency matrix of the network, either as raw counts or as proportions.
    _create_transaction()
        Internal method to create a random transaction between banks based on predefined probabilities.
    _random_bank(prob: np.ndarray) -> int
        Selects a bank for initiating a transaction based on a weighted probability distribution.
    _payment_link(sender: int, receiver: int)
        Establishes or updates a transaction link between two banks.
    """

    def __init__(self, total_banks: int, total_payments: int) -> None:
        """Initializes the payment network with the specified number of banks and payments.

        Parameters
        ----------
        total_banks : int
            The total number of banks included in the network simulation.
        total_payments : int
            The total number of payments to be simulated across the network.
        """
        self.G: nx.Graph = None
        self.total_banks = total_banks
        self.total_payments = total_payments

    @abc.abstractmethod
    def simulate_payments(self):
        """Abstract method to simulate payments between banks in the network.

        Must be implemented by all subclasses to define specific simulation strategies.
        """
        pass

    def extract_link_matrix(self, prop: bool = True) -> np.ndarray:
        """Retrieves the adjacency matrix of the network, showing transaction links between banks.

        Parameters
        ----------
        prop : bool, optional
            Determines whether the matrix should show proportions of total transactions (True)
            or raw transaction counts (False). Defaults to True.

        Returns
        -------
        np.ndarray
            The adjacency matrix of the network.

        Notes
        -----
        - If `prop` is True, the matrix entries represent the proportion of total transactions.
        - If `prop` is False, the matrix entries represent raw transaction counts.
        """
        matrix = nx.to_numpy_array(self.G, weight="weight", dtype=int)
        if not prop:
            return matrix
        return matrix / matrix.sum()

    def _create_transaction(self):
        """Internally generates a random transaction by selecting a sender and receiver from the network.

        The selection is influenced by the preferential attachment vector `h`. After selecting the banks,
        it updates the payment link and adjusts the preferential attachment strengths.

        Process
        -------
        1. Calculate the probability distribution for selecting banks based on `h`.
        2. Randomly select a sender and receiver using this probability distribution.
        3. Ensure that self-loop transactions are only allowed if `allow_self_loop` is True.
        4. Update the payment link between the sender and receiver.
        5. Increase the preferential attachment strength for both banks by `alpha`.
        """
        # select sender and receiver
        prob = self.h / self.h.sum()
        sender = self._random_bank(prob)
        receiver = self._random_bank(prob)

        # prevent self-loop transactions unless explicitly allowed
        while sender == receiver and not self.allow_self_loop:
            receiver = self._random_bank(prob)

        # update payment link between banks
        self._payment_link(sender, receiver)

        # update preferential attachment strength
        self.h[sender] += self.alpha
        self.h[receiver] += self.alpha

    def _random_bank(self, prob) -> int:
        """Randomly selects a bank to participate in a transaction, using a weighted probability distribution.

        Parameters
        ----------
        prob : np.ndarray
            An array of probabilities for each bank, indicating the likelihood of each bank being selected.

        Returns
        -------
        int
            The identifier of the bank selected.

        Notes
        -----
        - Uses `np.random.choice` to select a bank node based on the provided probability distribution.
        """
        return np.random.choice(self.G.nodes(), p=prob)

    def _payment_link(self, sender: int, receiver: int) -> None:
        """Creates or updates a payment link between two banks.

        Parameters
        ----------
        sender : int
            The identifier of the bank initiating the payment.
        receiver : int
            The identifier of the bank receiving the payment.

        Notes
        -----
        - If an edge between the sender and receiver already exists, increments its weight by 1.
        - If no edge exists, creates a new edge with an initial weight of 1.
        """
        if self.G.has_edge(sender, receiver):
            self.G[sender][receiver]["weight"] += 1
        else:
            self.G.add_edge(sender, receiver, weight=1)


class SimplePaymentNetwork(AbstractPaymentNetwork):
    """A simple implementation of the `AbstractPaymentNetwork` for simulating payments between banks.

    This class simulates a payment network where banks are added incrementally, and transactions occur based on
    a preferential attachment mechanism. It allows for the adjustment of parameters such as the number of banks,
    total payments, preferential attachment strength, and whether self-loops are allowed.

    Attributes
    ----------
    alpha : float
        The preferential attachment strength increment after each transaction.
    allow_self_loop : bool
        Determines whether transactions from a bank to itself are permitted.
    h : np.ndarray
        An array representing the preferential attachment strength for each bank.

    Methods
    -------
    simulate_payments(init_banks=2, increment=1)
        Simulates the payment processing between banks, gradually increasing the number of participating banks
        according to specified parameters.
    """

    def __init__(
        self,
        total_banks: int,
        total_payments: int,
        alpha: float = 1,
        allow_self_loop: bool = False,
    ) -> None:
        """Initializes a simple payment network simulation with specified parameters.

        Parameters
        ----------
        total_banks : int
            The total number of banks included in the network simulation.
        total_payments : int
            The total number of payments to be simulated across the network.
        alpha : float, optional
            The preferential attachment strength increment after each transaction, by default 1.0.
        allow_self_loop : bool, optional
            Determines whether transactions from a bank to itself are permitted, by default False.

        Raises
        ------
        ValueError
            If `total_banks` or `total_payments` is not a positive integer.
        """
        is_positive_int(total_banks, "total_banks")
        is_positive_int(total_payments, "total_payments")

        super().__init__(total_banks=total_banks, total_payments=total_payments)

        # set simulation parameters
        self.alpha = alpha
        self.allow_self_loop = allow_self_loop

    def simulate_payments(self, init_banks: int = 2, increment: int = 1) -> None:
        """Simulates the payment processing between banks.

        The simulation starts with an initial set of banks and incrementally adds
        more until all banks are active. Transactions are simulated based on a preferential
        attachment mechanism.

        Parameters
        ----------
        init_banks : int, optional
            The number of banks to start the simulation with, by default 2.
        increment : int, optional
            The maximum number of new banks to add in each iteration, by default 1.

        Raises
        ------
        ValueError
            If `init_banks` or `increment` is not a positive integer.

        Notes
        -----
        - The simulation dynamically adjusts the number of transactions based on the changing number of banks,
          maintaining the average payments per bank.
        - This process continues until all specified banks are included in the network.
        """
        is_positive_int(init_banks, "init_banks")
        is_positive_int(increment, "increment")

        # Initialize graph and preference vector
        self.G = nx.DiGraph()
        self.h = np.array([], dtype=float)

        # Set the initial numbers of payments and banks
        avg_payments = int(np.ceil(self.total_payments / (self.total_banks - init_banks)))
        add_payments = avg_payments
        add_banks = init_banks

        # Set counter for the number of banks and payments
        n_payments = 0
        n_banks = 0

        # Simulate transactions
        while add_banks > 0:
            # Initialize the next bank/node
            self.h = np.append(self.h, np.ones(add_banks))
            self.G.add_nodes_from(list(range(n_banks, n_banks + add_banks)))
            n_banks += add_banks

            # Simulate transactions
            for _ in range(add_payments):
                self._create_transaction()
            n_payments += add_payments

            # Determine the number of new banks and payments for the next iteration
            add_banks = min(randint(1, increment + 1), self.total_banks - n_banks)
            add_payments = min(avg_payments * add_banks, self.total_payments - n_payments)