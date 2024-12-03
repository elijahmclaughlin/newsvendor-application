# For this application, I will be using Streamlit
import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

# Making the app pretty
# title
st.title("Newsvendor Application")
st.markdown("""This app helps you optimize inventory decisions by using Monte Carlo simulation""")

# inputs
st.sidebar.header("Input Parameters")

random_seed = st.sidebar.number_input(
    "Random Seed",
    min_value=0,
    max_value=999999,
    value=1234,
    step=1,
    help="Set a random seed for reproducibility (optional)."
)

distribution_type = st.sidebar.selectbox(
    "Select Demand Distribution",
    options=["Normal", "Log-normal", "Poisson"]
)

mean_demand = st.sidebar.slider("Mean Demand", min_value=0, max_value=500, value=100, step=1,help="Expected average demand for the product")
std_dev_demand = st.sidebar.slider("Demand Std Dev", min_value=0, max_value=100, value=20, step=1,help="Expected standard deviation demand for the product")
unit_cost = st.sidebar.slider("Unit Cost", min_value=0.0, max_value=20.0, value=5.0, step=0.1,help="Expected unit cost for the product")
selling_price = st.sidebar.slider("Selling Price", min_value=0.0, max_value=50.0, value=10.0, step=0.1,help="Selling price for the product")
salvage_value = st.sidebar.slider("Salvage Value", min_value=0.0, max_value=20.0, value=2.0, step=0.1,help="Selling price for the product")
shortage_cost = st.sidebar.slider("Shortage Cost", min_value=0.0, max_value=10.0, value=1.0, step=0.1,help="Expected shortage cost for the product")
order_quantity = st.sidebar.slider("Order Quantity (Q)", min_value=1, max_value=500, value=100, step=1,help="Order quantity for the product")
simulations = st.sidebar.slider("Simulations", min_value=1000, max_value=100000, value=1000, step=100,help="Number of simulations")

st.write(f"**Selected Demand Distribution:** {distribution_type}")

# Monte Carlo Simulation - setting a seed for reproducibility
np.random.seed(random_seed)
if distribution_type == "Normal":
    demand_samples = np.random.normal(mean_demand, std_dev_demand, simulations)
elif distribution_type == "Log-normal":
    sigma = std_dev_demand / mean_demand
    mu = np.log(mean_demand) - 0.5 * sigma**2
    demand_samples = np.random.lognormal(mu, sigma, simulations)
elif distribution_type == "Poisson":
    demand_samples = np.random.poisson(mean_demand, simulations)
    
# Profit computation
sold = np.minimum(order_quantity, demand_samples)
unsold = np.maximum(0, order_quantity - demand_samples)
unmet_demand = np.maximum(0, demand_samples - order_quantity)

profits = (
    selling_price * sold
    - unit_cost * order_quantity
    + salvage_value * unsold
    - shortage_cost * unmet_demand
)

# Optimal Order Quantity Calculation
q_range = np.arange(1, 2 * int(mean_demand))
max_profit = float('-inf')
optimal_quantity = None

for q in q_range:
    sold_q = np.minimum(q, demand_samples)
    unsold_q = np.maximum(0, q - demand_samples)
    unmet_demand_q = np.maximum(0, demand_samples - q)

    q_profits = (
        selling_price * sold_q
        - unit_cost * q
        + salvage_value * unsold_q
        - shortage_cost * unmet_demand_q
    )

    avg_profit = np.mean(q_profits)
    if avg_profit > max_profit:
        max_profit = avg_profit
        optimal_quantity = q

# Metric calculation
expected_profit = np.mean(profits)
risk = np.std(profits)

st.write(f"**Expected Profit:** ${expected_profit:.2f}")
st.write(f"**Risk (Std Dev of Profit):** ${risk:.2f}")
st.write(f"**Optimal Inventory Units (U):** {optimal_quantity}")
st.write(f"**Maximum Expected Profit:** ${max_profit:.2f}")

# Profit histogram
profits_df = pd.DataFrame({"Profit": profits})
profit_chart = alt.Chart(profits_df).mark_bar().encode(
    alt.X("Profit", bin=alt.Bin(maxbins=30), title="Profit"),
    alt.Y('count()', title="Frequency")
).properties(
    title="Profit Distribution"
)
st.altair_chart(profit_chart, use_container_width=True)

# Demand Distribution histogram
demand_df = pd.DataFrame({"Demand": demand_samples})
demand_chart = alt.Chart(demand_df).mark_bar().encode(
    alt.X("Demand", bin=alt.Bin(maxbins=30), title="Demand"),
    alt.Y('count()', title="Frequency")
).properties(
    title=f"{distribution_type} Demand Distribution"
)
st.altair_chart(demand_chart, use_container_width=True)
