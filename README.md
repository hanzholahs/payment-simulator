# RTGS Anomaly Detection

## Overview

This repository contains the essential code and analysis for anomaly detection for Real-Time Gross Settlement (RTGS) systems. The notebooks included in this repository aim to simulate, analyze, and detect transaction data as well as anomalies within RTGS transactions, focusing on enhancing the robustness and reliability of financial transactions. Specificaly, there are three main analysis:

* Syntethic Data Generation: This analysis outlines the process of synthetic data generation that mimics typical RTGS transactions. 
* Bank Run Simulation: A bank run simulation mechanism is provided in the analysis, representing a rare crisis event for an RTGS system. The analysis serves as the foundation for testing our anomaly detection algorithms by providing a controlled environment with known parameters.
* Anomaly Detection Model: This analysis integrates a bank run simulation with anomaly detection techniques. The anomaly detection is based on Autoencoder model, leveraging a deep learning model to learn patterns in transaction flow data. It demonstrates how potential crises can be identified and mitigated in a simulated RTGS environment, using advanced analytical methods.

## Contact

For any queries regarding the use or development of these analysis, please raise an issue in the repository or contact the authors directly:

* Hanzholah Shobri (hanzholahs@gmail.com)
* Farhan M Sumadiredja (farhansumadiredja@gmail.com)
