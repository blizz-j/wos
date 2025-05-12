import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Streamlit app title
st.title("Whiteout Survival Refinement Simulator")

# Define tier data
tiers = [
    {"cost": 20, "rfc_probs": {1: 0.65, 2: 0.25, 3: 0.10}, "refinements": 20},  # Tier 1
    {"cost": 50, "rfc_probs": {2: 0.85, 3: 0.15}, "refinements": 20},         # Tier 2
    {"cost": 100, "rfc_probs": {3: 0.85, 4: 0.125, 5: 0.02, 6: 0.005}, "refinements": 20},  # Tier 3
    {"cost": 130, "rfc_probs": {3: 0.75, 4: 0.15, 5: 0.05, 6: 0.03, 7: 0.01, 8: 0.005, 9: 0.005}, "refinements": 20},  # Tier 4
    {"cost": 160, "rfc_probs": {3: 0.7, 4: 0.12, 5: 0.09, 6: 0.04, 7: 0.015, 8: 0.01, 9: 0.01, 10: 0.005, 11: 0.005, 12: 0.005}, "refinements": 20},  # Tier 5
]

# User inputs
st.sidebar.header("Simulation Parameters")
num_weeks = st.sidebar.slider("Number of Weeks", min_value=1, max_value=52, value=1)
daily_refinements = st.sidebar.slider("Refinements per Day", min_value=1, max_value=50, value=5)
num_simulations = 10000  # Fixed number of Monte Carlo iterations

# Function to simulate one refinement
def simulate_refinement(tier):
    rfc_values = list(tier["rfc_probs"].keys())
    probabilities = list(tier["rfc_probs"].values())
    return np.random.choice(rfc_values, p=probabilities)

# Function to run one simulation for given weeks and daily refinements
def run_simulation(num_weeks, daily_refinements):
    total_fc = 0
    total_rfc = 0
    current_tier = 0
    refinements_in_tier = 0

    for week in range(num_weeks):
        for day in range(7):  # 7 days in a week
            if refinements_in_tier >= 100:  # Weekly limit
                break
            for refinement in range(daily_refinements):
                if refinements_in_tier >= 100:  # Weekly limit
                    break
                # Check if we need to move to the next tier
                if refinements_in_tier >= tiers[current_tier]["refinements"]:
                    current_tier += 1
                    refinements_in_tier = 0
                    if current_tier >= len(tiers):
                        current_tier = len(tiers) - 1  # Stay at last tier
                # Apply daily discount for first refinement
                cost = tiers[current_tier]["cost"] / 2 if refinement == 0 else tiers[current_tier]["cost"]
                total_fc += cost
                # Simulate RFC yield
                total_rfc += simulate_refinement(tiers[current_tier])
                refinements_in_tier += 1

    return total_fc, total_rfc

# Run Monte Carlo simulation
rfc_results = []
for _ in range(num_simulations):
    fc, rfc = run_simulation(num_weeks, daily_refinements)
    rfc_results.append(rfc)

# Calculate FC consumed (same for all simulations)
total_fc_consumed = fc  # From the last simulation, as FC is deterministic

# Display results
st.header("Simulation Results")
st.write(f"**Total Fire Crystals Consumed**: {total_fc_consumed:.0f} FC")
st.write(f"**Number of Simulations**: {num_simulations}")
st.write(f"**Average Refined Fire Crystals**: {np.mean(rfc_results):.2f} RFC")
st.write(f"**Standard Deviation of Refined Fire Crystals**: {np.std(rfc_results):.2f} RFC")

# Plot histogram
fig, ax = plt.subplots()
ax.hist(rfc_results, bins=30, edgecolor="black")
ax.set_xlabel("Refined Fire Crystals (RFC)")
ax.set_ylabel("Frequency")
ax.set_title("Histogram of Refined Fire Crystals")
st.pyplot(fig)

# Instructions
st.sidebar.header("How to Use")
st.sidebar.write("""
- Adjust the **Number of Weeks** to simulate over multiple weeks.
- Set the **Refinements per Day** to define how many refinements are done daily.
- The app runs 10,000 simulations to generate a histogram of possible RFC outcomes.
- The total FC consumed is displayed along with the histogram of RFC results.
""")
