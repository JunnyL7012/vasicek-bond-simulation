import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import norm

st.title("Vasicek Model: Interest Rate & Bond Pricing Simulation")

st.markdown("""
### ℹ️ About This Simulation

This project demonstrates a bond pricing simulation using the Vasicek interest rate model. The model assumes that short-term interest rates follow a mean-reverting stochastic process, which is commonly used in financial engineering to model interest rate dynamics.

By adjusting key parameters like long-term mean rate (mu), speed of reversion (kappa), and volatility (sigma), we simulate multiple paths of short rates over time. Based on these rates, the simulation calculates the prices of zero-coupon bonds with various maturities (tau).

This type of model is used by professionals in asset management, risk modeling, and pricing interest rate derivatives.
""")

st.sidebar.header("Simulation Parameters")
kappa = st.sidebar.number_input("Kappa", value=0.1)
mu = st.sidebar.number_input("Mu", value=0.05)
sigma = st.sidebar.number_input("Sigma", value=0.01)
r0 = st.sidebar.number_input("Initial Rate (r0)", value=0.03)
num_scenarios = st.sidebar.number_input("Number of Scenarios", value=10, step=1)
dt = st.sidebar.number_input("dt (years)", value=0.25)
t_term = st.sidebar.number_input("t term (years)", value=5, step=1)
dtau = st.sidebar.number_input("dtau (years)", value=0.5)
tau_term = st.sidebar.number_input("tau term (years)", value=5, step=1)

def generate_scenarios(kappa, mu, sigma, r0, num_scenarios, dt, t_term):
    num_steps = int(t_term / dt) + 1
    scenarios = np.zeros((num_scenarios, num_steps))
    scenarios[:, 0] = r0
    for n in range(num_scenarios):
        for i in range(1, num_steps):
            scenarios[n, i] = scenarios[n, i-1] + kappa * (mu - scenarios[n, i-1]) * dt + sigma * np.sqrt(dt) * norm.ppf(np.random.rand())
    return scenarios

scenarios = generate_scenarios(kappa, mu, sigma, r0, int(num_scenarios), dt, int(t_term))
time_points = np.arange(0, t_term + dt, dt)

def calculate_bond_prices(scenarios, kappa, mu, sigma, dt, dtau, tau_term):
    num_scenarios, num_steps = scenarios.shape
    tau_steps = int(tau_term / dtau) + 1
    bond_prices = np.zeros((num_scenarios, num_steps, tau_steps))
    for n in range(num_scenarios):
        for i in range(num_steps):
            for j in range(tau_steps):
                tau = j * dtau
                B = (1 - np.exp(-kappa * tau)) / kappa
                A = np.exp((mu - (sigma**2) / (2 * kappa**2)) * (B - tau) - (sigma**2) * (B**2) / (4 * kappa))
                bond_prices[n, i, j] = A * np.exp(-B * scenarios[n, i])
    return bond_prices

bond_prices = calculate_bond_prices(scenarios, kappa, mu, sigma, dt, dtau, tau_term)

output_data = []
for n in range(int(num_scenarios)):
    for i, t in enumerate(time_points):
        row = [n + 1, t, scenarios[n, i]]
        row.extend(bond_prices[n, i, :])
        output_data.append(row)

columns = ['Scen', 'Time', 'Short Rate'] + [f'Price Tau {j+1}' for j in range(int(tau_term / dtau) + 1)]
output_df = pd.DataFrame(output_data, columns=columns)

st.subheader("Sample Output (First 10 Rows)")
st.dataframe(output_df.head(10))

st.subheader("Short Rate Paths (First 5 Scenarios)")
fig, ax = plt.subplots()
for i in range(min(5, int(num_scenarios))):
    ax.plot(time_points, scenarios[i], label=f'Scenario {i+1}')
ax.set_xlabel("Time (Years)")
ax.set_ylabel("Short Rate")
ax.set_title("Interest Rate Evolution")
st.pyplot(fig)

st.download_button("Download Output CSV", output_df.to_csv(index=False), "vasicek_output.csv", "text/csv")
