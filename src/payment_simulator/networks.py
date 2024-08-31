from abc import ABC, abstractmethod

import networkx as nx
import numpy as np
from numpy.random import randint


class AbstractPaymentNetwork(ABC):
    def __init__(self, total_banks: int) -> None:
        """
        Initializes the abstract payment network with a graph attribute.
        """
        self.G: nx.Graph = None
        self.total_banks = total_banks

    @abstractmethod
    def simulate_payments(self, init_banks: int | None):
        """
        Abstract method to simulate payments.
        This method should be implemented by subclasses.
        """
        pass

    def extract_link_matrix(self, prop: bool = True) -> np.ndarray:
        """
        Extracts and returns the adjacency matrix of the network graph.

        Parameters
        ----------
        prop : bool, optional
            If True, returns the matrix as proportions of the total transactions. If False, returns raw transaction counts.

        Returns
        -------
        np.ndarray
            A numpy ndarray representing the adjacency matrix of the network graph.
        """
        matrix = nx.to_numpy_array(self.G, weight="weight")
        if not prop:
            return matrix
        return matrix / matrix.sum()

    def _create_transaction(self):
        """
        Creates a transaction between randomly selected banks in the network.
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
        Selects a bank for initiating a transaction based on a weighted probability distribution.

        Parameters
        ----------
        prob : np.ndarray
            Array of probabilities for each bank.

        Returns
        -------
        int
            The selected bank's identifier.
        """
        return np.random.choice(self.G.nodes(), p=prob)

    def _payment_link(self, sender: int, receiver: int) -> None:
        """
        Creates or updates a payment link between two banks in the simulation graph.

        Parameters
        ----------
        sender : int
            The identifier of the bank initiating the payment.
        receiver : int
            The identifier of the bank receiving the payment.
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
        Initializes the RTGS Simulator with specified parameters.

        Parameters
        ----------
        total_banks : int
            Total number of banks participating in the RTGS simulation.
        avg_payments : int
            Average number of payments each bank is expected to process during the simulation.
        alpha : float, optional
            A learning rate parameter that influences the strength of preferential attachment in the simulation.
        allow_self_loop : bool, optional
            Boolean indicating whether transactions within the same bank (self-loops) are allowed.
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
        Simulates the payment processing between banks for a given period, starting with an initial set of banks.

        Parameters
        ----------
        init_banks : int, optional
            Initial number of banks to start the simulation with. If None, it defaults to half of the total banks rounded up.
        increment : int, optional
            The number of banks to add in each iteration.
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


class GroupedPaymentNetwork(AbstractPaymentNetwork):
    def __init__(
        self,
        total_banks: int,
        bank_groups: list[int | float],
        avg_payments: int,
        alpha: float = 0,
        allow_self_loop: bool = False,
    ) -> None:
        """
        Initializes the RTGS Simulator with specified parameters.

        Parameters
        ----------
        total_banks : int
            Total number of banks participating in the RTGS simulation.
        bank_groups : list of int or float
            List representing the proportion or number of banks in each group.
        avg_payments : int
            Average number of payments each bank is expected to process during the simulation.
        alpha : float, optional
            A learning rate parameter that influences the strength of preferential attachment in the simulation.
        allow_self_loop : bool, optional
            Boolean indicating whether transactions within the same bank (self-loops) are allowed.
        """
        super().__init__()

        # set simulation parameters
        self.alpha = alpha
        self.total_banks = total_banks
        self.bank_groups = bank_groups
        self.avg_payments = avg_payments
        self.allow_self_loop = allow_self_loop

    def simulate_payments(
        self, init_banks: int | None = None, increment: int = 2
    ) -> None:
        """
        Simulates the payment processing between banks for a given period, starting with an initial set of banks.

        Parameters
        ----------
        init_banks : int, optional
            Initial number of banks to start the simulation with. If None, it defaults to half of the total banks rounded up.
        increment : int, optional
            The number of banks to add in each iteration.
        """

        groups_count = np.array(self.bank_groups)
        groups_count = np.round(
            self.total_banks * (groups_count / groups_count.sum())
        ).astype(int)
        groups_count[-1] += self.total_banks - groups_count.sum()

        n_nodes = 0

        if init_banks is None:
            init_banks = int(1 + np.ceil(groups_count[0] / 2))

        # Initialize the graph with some nodes
        self.G = nx.DiGraph()  # graph network

        # Initialize the preference vector
        self.h = np.ones(init_banks, dtype=float)

        # Track the number of nodes
        n_nodes = 0

        # Simulate payment network
        for group, cum_limit in enumerate(np.cumsum(groups_count)):
            # Add nodes until meet group cumulative limit
            while n_nodes <= cum_limit:
                # Determine the number of new banks to add in the next iteration
                if n_nodes == 0:
                    n_addition = init_banks
                else:
                    n_addition = np.minimum(randint(1, increment), cum_limit - n_nodes)
                    if n_addition <= 0:
                        break
                    # Update the preference vector
                    self.h = np.append(self.h, np.ones(n_addition))

                # Initialize the next bank/node
                new_nodes = list(range(n_nodes, n_nodes + n_addition))
                self.G.add_nodes_from(new_nodes, group=group)

                # Simulate transactions
                for _ in range(self.avg_payments * n_addition):
                    self._create_transaction()

                # Keep track of the number of nodes
                n_nodes += len(new_nodes)
