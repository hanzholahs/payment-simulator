# `payment_simulator`: Simulating Large-value Transactions Data

The RTGS Simulation Package is specifically crafted to simulate large-value transactions within Real-Time Gross Settlement (RTGS) systems. It provides a robust set of tools designed to assist researchers and central bankers in analyzing the performance and reliability of RTGS systems under a variety of scenarios. This package is ideal for assessing the robustness of systems and the behavior of their participants through both normal transaction flows and controlled anomalies.

This development was inspired by an analysis aimed at anomaly detection in RTGS systems. You can explore the foundational [analysis notebook](./docs/anomaly_detection.ipynb). The included notebook is designed to simulate, analyze, and detect both typical transactions and anomalies within RTGS transactions. The analysis focus on enhancing the robustness and reliability of financial transactions through three main types of analysis: (i) Synthetic Data Generation;  (ii) Bank Run Simulation; and (iii) Anomaly Detection Model.

This package is currently **under development**. For any inquiries regarding the use or further development of these analyses, please feel free to raise an issue in the repository or contact the authors directly.

* Hanzholah Shobri (hanzholahs@gmail.com)
* Farhan M Sumadiredja (farhansumadiredja@gmail.com)

<!-- ## Installation

```bash
$ pip install payments_imulator
``` -->

## Usage

```python
import payment_simulator as ps
from payment_simulator.anomaly import AnomalyGenerator
from payment_simulator.networks import SimplePaymentNetwork
from payment_simulator.utils import random_payment_timing, random_payment_value

sim_periods = list(range(15))

sim_params = {
    "open_time": "06:30:00",
    "close_time": "18:30:00",
    "value_fn": random_payment_value,
    "timing_fn": random_payment_timing,
}

payment_network = SimplePaymentNetwork(
    total_banks=10, avg_payments=15, alpha=0.01, allow_self_loop=False
)

anomaly_generator = AnomalyGenerator(
    anomaly_start=5,
    anomaly_end=10,
    prob_start=0.8,
    prob_end=1,
    lambda_start=0.5,
    lambda_end=2.5,
    rate=0.5,
)

anomaly_transactions = ps.AnomalyTransactionSim(
    sim_id=2,
    network=payment_network,
    anomaly=anomaly_generator,
    **sim_params
)

anomaly_transactions.run(sim_periods)

anomaly_transactions.get_payments_df()
```

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`payments_imulator` is licensed under the terms of the MIT license. Check the [LICENSE](./LICENSE) file for details.