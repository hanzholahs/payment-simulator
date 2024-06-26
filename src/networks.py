from abc import ABC, abstractmethod

import networkx as nx
import numpy as np


class AbstractPaymentNetwork(ABC):
    def __init__(self) -> None:
        """
        Initializes the abstract payment network with a graph attribute.
        """
        self.G: nx.DiGraph = None

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
        alpha: float = 1.0,
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
        super().__init__()

        # set simulation parameters
        self.alpha = alpha
        self.total_banks = total_banks
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
        if init_banks is None:
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
                np.random.randint(1, increment), self.total_banks - len(self.G.nodes)
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
        alpha: float = 1.0,
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
        bank_groups_count = np.array(self.bank_groups)
        bank_groups_count = np.round(
            self.total_banks * (bank_groups_count / bank_groups_count.sum())
        ).astype(int)
        bank_groups_count[-1] += self.total_banks - bank_groups_count.sum()

        if init_banks is None:
            init_banks = int(1 + np.ceil(bank_groups_count[0] / 2))

        # Initialize the graph with some nodes
        self.G = nx.DiGraph()  # graph network
        self.G.add_nodes_from(list(range(init_banks)), group=0)

        # Initialize preference vector
        self.h = np.ones(init_banks, dtype=float)

        # Set number of payments for the iteration
        n_payments = self.avg_payments * init_banks

        # Simulate payment network
        for group, cumulative_limit in enumerate(np.cumsum(bank_groups_count)):
            while len(self.G.nodes) <= cumulative_limit:
                # Simulate transactions
                for _ in range(n_payments):
                    self._create_transaction()

                # Determine the number of new banks to add in the next iteration
                n_addition = np.minimum(
                    np.random.randint(1, increment),
                    cumulative_limit - len(self.G.nodes),
                )
                if n_addition <= 0:
                    break

                # Initialize the next bank/node
                new_nodes = list(
                    range(len(self.G.nodes), len(self.G.nodes) + n_addition)
                )
                self.G.add_nodes_from(new_nodes, group=group)

                # Update the preference vector
                self.h = np.append(self.h, np.ones(n_addition))

                # Update the number of payments for the next iteration
                n_payments = self.avg_payments * n_addition


if __name__ == "__main__":
    init_banks = 3
    increment = 4
    total_banks = 9
    alpha = 0.00001
    avg_payments = 1000
    allow_self_loop = False

    network = SimplePaymentNetwork(
        total_banks=total_banks,
        avg_payments=avg_payments,
        alpha=alpha,
        allow_self_loop=allow_self_loop,
    )
    network.simulate_payments(increment=increment, init_banks=init_banks)

    print("Bank strengths:")
    print(network.h)

    print("Network links:")
    print(np.round(network.extract_link_matrix(), 4))
    print(network.extract_link_matrix(False).sum() == total_banks * avg_payments)

    g_network = GroupedPaymentNetwork(
        total_banks=25,
        bank_groups=[10, 35, 23, 42],
        avg_payments=100,
        alpha=0.01,
        allow_self_loop=False,
    )
    g_network.simulate_payments(increment=25, init_banks=5)
    print("Network links:")
    print(np.round(g_network.extract_link_matrix(), 4))
