

import numpy as np
import pandas as pd
from scipy.stats import norm

# Step 1: Read Parameters from CSV
params = pd.read_csv('FNCE5323_24Sm_Asgn1-Params.csv', index_col=0, header=None)
params = params.squeeze().to_dict()
kappa = float(params['Kappa ='])
mu = float(params['Mu ='])
sigma = float(params['Sigma ='])
r0 = float(params['r0 ='])
num_scenarios = int(params['Nmb Scens ='])
dt = float(params['dt (in years) ='])
t_term = int(params['t term (in years) ='])
dtau = float(params['dtau (in years) ='])
tau_term = int(params['tau term (in years) ='])

# Step 2: Generate Stochastic Scenarios for Interest Rates
def generate_scenarios(kappa, mu, sigma, r0, num_scenarios, dt, t_term):
    num_steps = int(t_term / dt) + 1
    scenarios = np.zeros((num_scenarios, num_steps))
    scenarios[:, 0] = r0
    for n in range(num_scenarios):
        for i in range(1, num_steps):
            dt_sqrt = np.sqrt(dt)
            scenarios[n, i] = scenarios[n, i-1] + kappa * (mu - scenarios[n, i-1]) * dt + sigma * dt_sqrt * norm.ppf(np.random.rand())
    return scenarios

scenarios = generate_scenarios(kappa, mu, sigma, r0, num_scenarios, dt, t_term)

# Step 3: Calculate Zero-Coupon Bond Prices
def calculate_bond_prices(scenarios, kappa, mu, sigma, dt, dtau, tau_term):
    num_scenarios, num_steps = scenarios.shape
    tau_steps = int(tau_term / dtau) + 1
    bond_prices = np.zeros((num_scenarios, num_steps, tau_steps))
    
    for n in range(num_scenarios):
        for i in range(num_steps):
            for j in range(tau_steps):
                t = i * dt
                tau = j * dtau
                B = (1 - np.exp(-kappa * tau)) / kappa
                A = np.exp((mu - (sigma**2) / (2 * kappa**2)) * (B - tau) - (sigma**2) * (B**2) / (4 * kappa))
                bond_prices[n, i, j] = A * np.exp(-B * scenarios[n, i])
    return bond_prices

bond_prices = calculate_bond_prices(scenarios, kappa, mu, sigma, dt, dtau, tau_term)

# Step 4: Prepare Output Data
output_data = []
time_points = np.arange(0, t_term + dt, dt)

for n in range(num_scenarios):
    for i, t in enumerate(time_points):
        row = [n + 1, t, scenarios[n, i]]
        row.extend(bond_prices[n, i, :])
        output_data.append(row)

columns = ['Scen', 'Time', 'Short Rate'] + [f'Price Tau {j + 1}' for j in range(int(tau_term / dtau) + 1)]
output_df = pd.DataFrame(output_data, columns=columns)

# Step 5: Write Output to CSV
output_df.to_csv('Asgn1_Out_H4.csv', index=False)

# Optional: Verify Output
print(output_df.head())
