import numpy as np
import pandas as pd
import os
import json

# Define the SystemModel class
class SystemModel:
    def __init__(self, D, S_max, S_0, k_s, k_e, E_max, eta_in, eta_out, P_max, cost_op, cost_st, cost_ens, cost_loss):
        self.D = D  # Demand
        self.S_max = S_max  # Maximum storage capacity
        self.S_0 = S_0  # Initial storage level
        self.k_s = k_s  # Storage loss coefficient
        self.k_e = k_e  # Energy loss coefficient (not used in this model)
        self.E_max = E_max  # Maximum energy input
        self.eta_in = eta_in  # Input efficiency
        self.eta_out = eta_out  # Output efficiency
        self.P_max = P_max  # Maximum power output
        self.cost_op = cost_op  # Operational cost
        self.cost_st = cost_st  # Storage cost
        self.cost_ens = cost_ens  # Energy not served cost
        self.cost_loss = cost_loss  # Energy loss cost

    def simulate(self, E_in, T):
        S = np.zeros(T + 1)
        S[0] = self.S_0
        P_out = np.zeros(T)
        L_s = np.zeros(T)
        L_e = np.zeros(T)  # Not used
        ENS = np.zeros(T)
        Q = np.zeros(T)  # Overflow

        for t in range(T):
            # Storage update
            S_potential = S[t] + E_in[t] * self.eta_in - self.D[t] / self.eta_out

            # Storage loss
            L_s[t] = S[t] * self.k_s

            # Energy available for output
            E_avail = S[t] - L_s[t]

            # Actual power output
            if E_avail * self.eta_out >= self.D[t]:
                P_out[t] = self.D[t]
                ENS[t] = 0
            else:
                P_out[t] = E_avail * self.eta_out
                ENS[t] = self.D[t] - P_out[t]

            # Update storage level
            S[t+1] = S[t] + E_in[t] * self.eta_in - P_out[t] / self.eta_out - L_s[t]

            # Check for overflow
            if S[t+1] > self.S_max:
                Q[t] = S[t+1] - self.S_max
                S[t+1] = self.S_max
            else:
                Q[t] = 0

        return {
            'storage_level': S[:-1],
            'power_output': P_out,
            'storage_loss': L_s,
            'energy_loss': L_e, # Should be L_e[t] if used
            'energy_not_served': ENS,
            'overflow': Q,
            'time': np.arange(T)
        }

# Define the generate_synthetic_data function
def generate_synthetic_data(scenario='support'):
    np.random.seed(42)  # for reproducibility
    T = 100  # Time horizon
    time_steps = np.arange(T)

    # Define system parameters (can be varied for different scenarios)
    if scenario == 'support':
        D = np.random.uniform(50, 150, T)  # Demand
        E_max = 200
    elif scenario == 'fail':
        D = np.random.uniform(150, 250, T) # Higher demand
        E_max = 100 # Lower energy max
    elif scenario == 'marginal':
        D = np.random.uniform(100, 200, T)
        E_max = 150
    else:
        raise ValueError("Unknown scenario type")

    S_max = 1000
    S_0 = 500
    k_s = 0.01
    k_e = 0.01 # not used currently
    eta_in = 0.9
    eta_out = 0.8
    P_max = 200 # Max power output, ensure it can meet peak D with some buffer if not failing
    cost_op = 0.1
    cost_st = 0.05
    cost_ens = 10
    cost_loss = 1 # Cost for energy lost (overflow or storage loss)

    system_model = SystemModel(D, S_max, S_0, k_s, k_e, E_max, eta_in, eta_out, P_max, cost_op, cost_st, cost_ens, cost_loss)

    # Generate synthetic flow inputs (energy inputs)
    # For simplicity, let's assume flow_inputs are somewhat correlated with demand,
    # but with some randomness and constrained by E_max
    base_flow_input = D / eta_in # Ideal input to meet demand
    noise = np.random.normal(0, 20, T)
    flow_inputs = np.clip(base_flow_input + noise, 0, E_max)

    # Simulate to get true data (observed_data)
    true_data = system_model.simulate(flow_inputs, T)

    # Add some noise to observed_data to make it more realistic for calibration
    observed_data = {}
    for key, value in true_data.items():
        if key != 'time': # No noise for time array
            noise_factor = np.random.normal(1, 0.05, value.shape) # 5% noise
            observed_data[key] = value * noise_factor
        else:
            observed_data[key] = value

    # Ensure 'energy_input' is part of observed_data for consistency with calibration
    observed_data['energy_input'] = flow_inputs

    return flow_inputs, time_steps, observed_data

if __name__ == "__main__":
    # Create test_data directory if it doesn't exist
    if not os.path.exists("test_data"):
        os.makedirs("test_data")

    scenarios = ['support', 'fail', 'marginal']

    for scenario in scenarios:
        flow_inputs, time_steps, observed_data_np = generate_synthetic_data(scenario)

        # Structure data for JSON serialization (convert numpy arrays to lists)
        data_to_save = {
            "flow_inputs": flow_inputs.tolist(),
            "time_steps": time_steps.tolist(),
            "observed_data": {
                "storage_level": observed_data_np['storage_level'].tolist(),
                "power_output": observed_data_np['power_output'].tolist(),
                "storage_loss": observed_data_np['storage_loss'].tolist(),
                "energy_loss": observed_data_np['energy_loss'].tolist(),
                "energy_not_served": observed_data_np['energy_not_served'].tolist(),
                "overflow": observed_data_np['overflow'].tolist(),
                "time": observed_data_np['time'].tolist(),
                "energy_input": observed_data_np['energy_input'].tolist()
            }
        }

        output_filename = f"test_data/{scenario}_data.json"
        with open(output_filename, 'w') as f:
            json.dump(data_to_save, f, indent=4)

        print(f"Saved data for scenario '{scenario}' to {output_filename}")
