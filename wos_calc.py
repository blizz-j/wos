import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define tier data
tiers = [
    {"cost": 20, "rfc_probs": {1: 0.65, 2: 0.25, 3: 0.10}, "refinements": 20},  # Tier 1
    {"cost": 50, "rfc_probs": {2: 0.85, 3: 0.15}, "refinements": 20},         # Tier 2
    {"cost": 100, "rfc_probs": {3: 0.85, 4: 0.125, 5: 0.02, 6: 0.005}, "refinements": 20},  # Tier 3
    {"cost": 130, "rfc_probs": {3: 0.75, 4: 0.15, 5: 0.05, 6: 0.03, 7: 0.01, 8: 0.005, 9: 0.005}, "refinements": 20},  # Tier 4
    {"cost": 160, "rfc_probs": {3: 0.7, 4: 0.12, 5: 0.09, 6: 0.04, 7: 0.015, 8: 0.01, 9: 0.01, 10: 0.005, 11: 0.005, 12: 0.005}, "refinements": 20},  # Tier 5
]

# Function to simulate one refinement
def simulate_refinement(tier):
    rfc_values = list(tier["rfc_probs"].keys())
    probabilities = list(tier["rfc_probs"].values())
    return np.random.choice(rfc_values, p=probabilities)

# Function to run one simulation
def run_simulation(weeks, day1_refinements, days2_7_refinements):
    total_fc = 0
    total_rfc = 0
    for _ in range(weeks):
        current_tier = 0
        refinements_in_tier = 0
        # Simulate each day of the week
        for day in range(7):
            if day == 0:
                refinements_today = day1_refinements
            else:
                refinements_today = days2_7_refinements
            for refinement in range(refinements_today):
                # Apply daily discount to first refinement
                if refinement == 0:
                    cost = tiers[current_tier]["cost"] / 2
                else:
                    cost = tiers[current_tier]["cost"]
                total_fc += cost
                total_rfc += simulate_refinement(tiers[current_tier])
                refinements_in_tier += 1
                # Tier progression
                if refinements_in_tier >= tiers[current_tier]["refinements"]:
                    current_tier += 1
                    refinements_in_tier = 0
                    if current_tier >= len(tiers):
                        current_tier = len(tiers) - 1  # Cap at highest tier
    return total_fc, total_rfc

# Streamlit app
st.title("Whiteout Survival Calculators")

# Sidebar menu for calculator selection
calculator = st.sidebar.selectbox("Select Calculator", ["Refinement Simulator"])

if calculator == "Refinement Simulator":
    # Horizontal tabs
    tab1, = st.tabs(["Refinement Simulator"])
    
    with tab1:
        st.header("Refinement Simulator")
        
        # Input UI elements
        weeks = st.number_input("Number of Weeks", min_value=1, max_value=52, value=1)
        day1_refinements = st.number_input("FC Refined on Day 1", min_value=1, max_value=100, value=1)
        max_days2_7 = (100 - day1_refinements) // 6
        days2_7_refinements = st.number_input(
            f"FC Refined on Days 2-7 (max {max_days2_7})",
            min_value=0,
            max_value=max_days2_7,
            value=0
        )
        simulations = st.number_input("Number of Monte Carlo Simulations", min_value=1, max_value=100000, value=10000)
        
        # Run simulations button
        if st.button("Run Simulations"):
            # Perform Monte Carlo simulations
            rfc_results = []
            for _ in range(simulations):
                fc, rfc = run_simulation(weeks, day1_refinements, days2_7_refinements)
                rfc_results.append(rfc)
            total_fc = fc  # FC consumed is deterministic
            
            # Calculate summary statistics
            min_rfc = min(rfc_results)
            avg_rfc = np.mean(rfc_results)
            
            # Calculate deciles
            deciles = np.percentile(rfc_results, [10, 20, 30, 40, 50, 60, 70, 80, 90])
            
            # Build decile table
            decile_data = {
                "Decile": [f"{i*10}%" for i in range(1, 10)],
                "FC Spent": [total_fc] * 9,
                "rFC Generated": deciles,
                "FC/rFC Ratio": [total_fc / rfc if rfc > 0 else float('inf') for rfc in deciles]
            }
            decile_df = pd.DataFrame(decile_data)
            
            # Display results
            st.subheader("Simulation Results")
            
            # Summary text output
            st.write(f"**FC Consumed**: {total_fc:.0f}")
            st.write(f"**Min rFC Gained**: {min_rfc:.0f}")
            st.write(f"**Average rFC Gained**: {avg_rfc:.2f}")
            
            # Decile breakdown table
            st.subheader("Breakdown by Decile")
            st.table(decile_df)
            
            # Histogram of rFC generated
            st.subheader("Distribution of rFC Generated")
            fig, ax = plt.subplots()
            ax.hist(rfc_results, bins=30, edgecolor="black")
            ax.set_xlabel("Refined Fire Crystals (rFC)")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
