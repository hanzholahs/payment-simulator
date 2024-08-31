from abc import ABC, abstractmethod

import networkx as nx
import numpy as np
from numpy.random import randint


class AbstractPaymentNetwork(ABC):
    """
    An abstract base class for creating payment networks that simulate transactions
    between banks within a financial system.

    Attributes
    ----------
    total_banks : int
        The total number of banks included in the network simulation.
    G : networkx.Graph
        The graph representing the payment network, where nodes represent banks
        and edges represent transactions between them.

    Methods
    -------
    simulate_payments(init_banks: int | None)
        Simulates transactions across the network. This method must be implemented by subclasses.
    extract_link_matrix(prop: bool = True) -> np.ndarray
        Returns the adjacency matrix of the network, either as raw counts or as proportions.
    _create_transaction()
        Internal method to create a random transaction between banks based on predefined probabilities.
    _random_bank(prob: np.ndarray) -> int
        Selects a bank for initiating a transaction based on a weighted probability distribution.
    _payment_link(sender: int, receiver: int)
        Establishes or updates a transaction link between two banks.
    """

    def __init__(self, total_banks: int) -> None:
        """
        Initializes the payment network with the specified number of banks.

        Parameters
        ----------
        total_banks : int
            Specifies the total number of banks to include in the network.
        """
        self.G: nx.Graph = None
        self.total_banks = total_banks

    @abstractmethod
    def simulate_payments(self, init_banks: int | None):
        """
        Abstract method to simulate payments between banks in the network.
        Must be implemented by all subclasses to define specific simulation strategies.

        Parameters
        ----------
        init_banks : int, optional
            The number of banks that start transacting at the initiation of the simulation.
        """
        pass

    def extract_link_matrix(self, prop: bool = True) -> np.ndarray:
        """
        Retrieves the adjacency matrix of the network, showing transaction links between banks.

        Parameters
        ----------
        prop : bool, optional
            Determines whether the matrix should show proportions of total transactions (True)
            or raw transaction counts (False). Defaults to True.

        Returns
        -------
        np.ndarray
            The adjacency matrix of the network.
        """
        matrix = nx.to_numpy_array(self.G, weight="weight")
        if not prop:
            return matrix
        return matrix / matrix.sum()

    def _create_transaction(self):
        """
        Internally generates a random transaction by selecting a sender and receiver from the network.
        The selection is influenced by the preferential attachment vector 'h'.
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
        """
        Randomly selects a bank to initiate a transaction, using a weighted probability distribution.

        Parameters
        ----------
        prob : np.ndarray
            An array of probabilities for each bank, indicating the likelihood of each bank initiating a transaction.

        Returns
        -------
        int
            The identifier of the bank selected to initiate the transaction.
        """
        return np.random.choice(self.G.nodes(), p=prob)

    def _payment_link(self, sender: int, receiver: int) -> None:
        """
        Creates or updates a payment link between two banks.

        Parameters
        ----------
        sender : int
            The identifier of the bank initiating the payment.
        receiver : int
            The identifier of the bank receiving the payment.

        Notes
        -----
        This method increments the weight of the edge between the sender and receiver to reflect
        the occurrence of a transaction.
        """
        if self.G.has_edge(sender, receiver):
            self.G[sender][receiver]["weight"] += 1
        else:
            self.G.add_edge(sender, receiver, weight=1)


class SimplePaymentNetwork(AbstractPaymentNetwork):
    def __init__(
        self,
        total_banks: int,
        avg_payments: int,
        alpha: float = 0,
        allow_self_loop: bool = False,
    ) -> None:
        """
        Initializes a simple payment network simulation, defining parameters such as the number of banks,
        the average number of transactions per bank, the strength of preferential attachment, and whether
        self-transactions (self-loops) are permitted.

        Parameters
        ----------
        total_banks : int
            The total number of banks participating in the simulation.
        avg_payments : int
            The average number of transactions that each bank is expected to process.
        alpha : float, optional
            The learning rate parameter that influences the strength of preferential attachment in the network,
            default is 0, which implies no preferential attachment is considered.
        allow_self_loop : bool, optional
            Indicates whether transactions within the same bank are allowed, default is False.
        """
        super().__init__(total_banks=total_banks)

        # set simulation parameters
        self.alpha = alpha
        self.avg_payments = avg_payments
        self.allow_self_loop = allow_self_loop

    def simulate_payments(
        self, init_banks: int | None = None, increment: int = 2
    ) -> None:
        """
        Simulates the payment processing between banks, gradually increasing the number of participating banks
        according to specified parameters. The simulation starts with an initial set of banks and incrementally adds
        more until all banks are active.

        Parameters
        ----------
        init_banks : int, optional
            The initial number of banks to include in the simulation. If not specified, it defaults to half of the total banks, rounded up.
        increment : int, optional
            The number of additional banks added in each step of the simulation. Must be a positive integer.

        Raises
        ------
        AssertionError
            If `increment` is not a positive integer.

        Notes
        -----
        The simulation dynamically adjusts the number of transactions based on the changing number of banks,
        maintaining the average payments per bank. This process continues until all specified banks are included
        in the network or the addition results in no new banks due to constraints.
        """
        assert increment > 0, "`increment` must be positive integer."

        if init_banks is None or init_banks < 2:
            init_banks = int(1 + np.ceil(self.total_banks / 2))

        # Initialize the graph with some nodes
        self.G = nx.DiGraph()  # graph network
        self.G.add_nodes_from(list(range(init_banks)))

        # Initialize preference vector
        self.h = np.ones(init_banks, dtype=float)

        # Set number of payments for the iteration
        n_payments = self.avg_payments * init_banks

        # Simulate payment network
        while len(self.G.nodes) <= self.total_banks:
            # Simulate transactions
            for _ in range(n_payments):
                self._create_transaction()

            # Determine the number of new banks to add in the next iteration
            n_addition = np.minimum(
                randint(1, increment), self.total_banks - len(self.G.nodes)
            )
            if n_addition <= 0:
                break

            # Initialize the next bank/node
            new_nodes = list(range(len(self.G.nodes), len(self.G.nodes) + n_addition))
            self.G.add_nodes_from(new_nodes)

            # Update the preference vector
            self.h = np.append(self.h, np.ones(n_addition))

            # Update the number of payments for the next iteration
            n_payments = self.avg_payments * n_addition
