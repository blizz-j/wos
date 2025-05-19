import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define tier data
tiers = [
    {"cost": 20, "rfc_probs": {1: 0.65, 2: 0.25, 3: 0.10}, "refinements": 20, "min_rfc": 1, "max_rfc": 3},  # Tier 1
    {"cost": 50, "rfc_probs": {2: 0.85, 3: 0.15}, "refinements": 20, "min_rfc": 2, "max_rfc": 3},       # Tier 2
    {"cost": 100, "rfc_probs": {3: 0.85, 4: 0.125, 5: 0.02, 6: 0.005}, "refinements": 20, "min_rfc": 3, "max_rfc": 6},  # Tier 3
    {"cost": 130, "rfc_probs": {3: 0.75, 4: 0.15, 5: 0.05, 6: 0.03, 7: 0.01, 8: 0.005, 9: 0.005}, "refinements": 20, "min_rfc": 3, "max_rfc": 9},  # Tier 4
    {"cost": 160, "rfc_probs": {3: 0.7, 4: 0.12, 5: 0.09, 6: 0.04, 7: 0.015, 8: 0.01, 9: 0.01, 10: 0.005, 11: 0.005, 12: 0.005}, "refinements": 20, "min_rfc": 3, "max_rfc": 12},  # Tier 5
]

# Function to calculate maximum possible rFC for given refinements
def calculate_max_rfc(refinements):
    max_rfc = 0
    remaining = refinements
    for tier in tiers:
        if remaining >= tier["refinements"]:
            max_rfc += tier["refinements"] * tier["max_rfc"]
            remaining -= tier["refinements"]
        else:
            max_rfc += remaining * tier["max_rfc"]
            break
    return max_rfc

# Function to simulate one refinement
def simulate_refinement(tier):
    rfc_values = list(tier["rfc_probs"].keys())
    probabilities = list(tier["rfc_probs"].values())
    return np.random.choice(rfc_values, p=probabilities)

# Function to run one simulation for refinement simulator
def run_simulation(weeks, day1_refinements, days2_7_refinements):
    total_fc = 0
    total_rfc = 0
    for _ in range(weeks):
        current_tier = 0
        refinements_in_tier = 0
        for day in range(7):
            refinements_today = day1_refinements if day == 0 else days2_7_refinements
            for refinement in range(refinements_today):
                cost = tiers[current_tier]["cost"] / 2 if refinement == 0 else tiers[current_tier]["cost"]
                total_fc += cost
                total_rfc += simulate_refinement(tiers[current_tier])
                refinements_in_tier += 1
                if refinements_in_tier >= tiers[current_tier]["refinements"]:
                    current_tier += 1
                    refinements_in_tier = 0
                    if current_tier >= len(tiers):
                        current_tier = len(tiers) - 1
    return total_fc, total_rfc

# Function to run one simulation for luck assessment
def run_luck_simulation(refinements):
    total_fc = 0
    total_rfc = 0
    current_tier = 0
    refinements_in_tier = 0
    remaining = refinements
    for day in range(7):
        if remaining == 0:
            break
        # First refinement with discount
        if refinements_in_tier >= tiers[current_tier]["refinements"]:
            current_tier += 1
            refinements_in_tier = 0
            if current_tier >= len(tiers):
                current_tier = len(tiers) - 1
        cost = tiers[current_tier]["cost"] / 2
        total_fc += cost
        total_rfc += simulate_refinement(tiers[current_tier])
        refinements_in_tier += 1
        remaining -= 1
        # Additional refinements without discount
        for _ in range(min(remaining, 100)):
            if refinements_in_tier >= tiers[current_tier]["refinements"]:
                current_tier += 1
                refinements_in_tier = 0
                if current_tier >= len(tiers):
                    current_tier = len(tiers) - 1
            cost = tiers[current_tier]["cost"]
            total_fc += cost
            total_rfc += simulate_refinement(tiers[current_tier])
            refinements_in_tier += 1
            remaining -= 1
            if remaining == 0:
                break
    return total_fc, total_rfc

# Function to calculate strategy metrics for summary table
def calculate_strategy_metrics(day1_refinements, days2_7_refinements):
    total_fc = 0
    total_min_rfc = 0
    total_expected_rfc = 0
    current_tier = 0
    refinements_in_tier = 0
    for day in range(7):
        refinements_today = day1_refinements if day == 0 else days2_7_refinements
        for refinement in range(refinements_today):
            cost = tiers[current_tier]["cost"] / 2 if refinement == 0 else tiers[current_tier]["cost"]
            total_fc += cost
            total_min_rfc += tiers[current_tier]["min_rfc"]
            expected_rfc = sum(k * v for k, v in tiers[current_tier]["rfc_probs"].items())
            total_expected_rfc += expected_rfc
            refinements_in_tier += 1
            if refinements_in_tier >= tiers[current_tier]["refinements"]:
                current_tier += 1
                refinements_in_tier = 0
                if current_tier >= len(tiers):
                    current_tier = len(tiers) - 1
    return total_fc, total_min_rfc, total_expected_rfc

# Streamlit app
# Sidebar menu for calculator selection
calculator = st.sidebar.selectbox("WOS Calculators", ["Refinement Simulator"])

if calculator == "Refinement Simulator":
    # Horizontal tabs with flipped order for 2nd and 3rd tabs
    tab1, tab2, tab3 = st.tabs(["Refinement Simulator", "Strategies", "Was I Lucky?"])
    
    with tab1:
        st.subheader("Refinement Simulator")
        
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
        simulations = st.number_input("Number of Monte Carlo Simulations", min_value=1, max_value=100000, value=4000)
        
        # Run simulations button
        if st.button("Run Simulations"):
            # Perform Monte Carlo simulations
            rfc_results = []
            for _ in range(simulations):
                fc, rfc = run_simulation(weeks, day1_refinements, days2_7_refinements)
                rfc_results.append(rfc)
            total_fc = fc  # FC consumed is deterministic
            
            # Calculate summary statistics
            percentile_5 = np.percentile(rfc_results, 5)  # 5th percentile (unlucky)
            percentile_50 = np.percentile(rfc_results, 50)  # 50th percentile (average)
            percentile_95 = np.percentile(rfc_results, 95)  # 95th percentile (very lucky)
            
            # Calculate quantiles
            quantiles = np.percentile(rfc_results, [1, 5, 10, 25, 50, 75, 90, 95, 99])
            
            # Build quantile table
            quantile_data = {
                "Quantile": ["1%", "5%", "10%", "25%", "50%", "75%", "90%", "95%", "99%"],
                "FC Spent": [int(total_fc)] * 9,
                "rFC Generated": [int(q) for q in quantiles],
                "FC/rFC Ratio": [round(total_fc / q, 2) if q > 0 else float('inf') for q in quantiles]
            }
            quantile_df = pd.DataFrame(quantile_data)
            
            # Style DataFrame with centered columns and 2 decimal places for ratio
            styled_df = quantile_df.style.set_properties(**{
                'text-align': 'center'
            }).set_table_styles([
                {'selector': 'th', 'props': [('text-align', 'center')]}
            ]).format({
                'FC/rFC Ratio': '{:.2f}'  # Ensure 2 decimal places
            })
            
            # Display results
            st.subheader(f"After {simulations} simulations..")
            
            # Summary text output
            st.write(f"**FC consumed by refinement**: {total_fc:.0f}")
            st.write(f"**95% chance to make at least**: {percentile_5:.0f} rFC (unlucky)")
            st.write(f"**Average rFC made**: {percentile_50:.0f} rFC (normal)")
            st.write(f"**5% chance to make**: {percentile_95:.0f} rFC (very lucky)")
            
            # Quantile breakdown table
            st.subheader("Breakdown by Quantile")
            st.dataframe(styled_df, use_container_width=True)
            
            # Histogram of rFC generated
            st.subheader("Distribution of rFC Generated")
            fig, ax = plt.subplots()
            ax.hist(rfc_results, bins=30, edgecolor="black")
            ax.set_xlabel("Refined Fire Crystals (rFC)")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
    
    with tab2:
        st.subheader("Summary of Strategies")
        
        # Define list of common strategies
        strategies = [
            {"description": "Slowest (f2p)", "day1_refinements": 1, "days2_7_refinements": 1},
            {"description": "Low spender or f2p", "day1_refinements": 20, "days2_7_refinements": 1},
            {"description": "Mid spender", "day1_refinements": 40, "days2_7_refinements": 1},
            {"description": "High spender", "day1_refinements": 60, "days2_7_refinements": 1},
            {"description": "Whale fast", "day1_refinements": 80, "days2_7_refinements": 1},
            {"description": "Max speed", "day1_refinements": 94, "days2_7_refinements": 1}
        ]
        
        # Calculate metrics for each strategy
        strategy_data = []
        for strategy in strategies:
            day1_refinements = strategy["day1_refinements"]
            days2_7_refinements = strategy["days2_7_refinements"]
            total_fc, min_rfc, expected_rfc = calculate_strategy_metrics(day1_refinements, days2_7_refinements)
            avg_fc_per_rfc = total_fc / expected_rfc if expected_rfc > 0 else float('inf')
            refinements_str = f"{day1_refinements}/" + "/".join([str(days2_7_refinements)] * 6)
            strategy_data.append({
                "Description": strategy["description"],
                "Refine day 1-7": refinements_str,
                "FC used/week": int(total_fc),
                "Min rFC/week": int(min_rfc),
                "Ave rFC/week": round(expected_rfc, 2),
                "Ave FC/rFC": round(avg_fc_per_rfc, 2)
            })
        
        # Create and style DataFrame
        strategy_df = pd.DataFrame(strategy_data)
        styled_strategy_df = strategy_df.style.set_properties(**{
            'text-align': 'center'
        }).set_table_styles([
            {'selector': 'th', 'props': [('text-align', 'center')]}
        ]).format({
            'Average rFC/week': '{:.2f}',
            'Average FC/rFC': '{:.2f}'
        })
        
        # Display the table
        st.dataframe(styled_strategy_df, use_container_width=True)
    
    with tab3:
        st.subheader("Was I Lucky?")
        
        # Input UI elements
        refinements = st.number_input("Refinements this Week", min_value=1, max_value=100, value=1)
        rfc_acquired = st.number_input("rFC Acquired", min_value=1, value=1, format="%d")
        
        # Assess luck button
        if st.button("Assess my Luck!"):
            # Validate rFC acquired
            max_rfc = calculate_max_rfc(refinements)
            if rfc_acquired > max_rfc:
                st.error(f"Not possible! rFC acquired ({rfc_acquired}) exceeds the theoretical maximum ({max_rfc}) for {refinements} refinements. Please double-check your inputs.")
            else:
                # Run 10,000 Monte Carlo simulations
                rfc_results = []
                total_fc = 0
                for _ in range(10000):
                    fc, rfc = run_luck_simulation(refinements)
                    rfc_results.append(rfc)
                    total_fc = fc  # FC is deterministic
                
                # Calculate percentile
                percentile = np.mean(np.array(rfc_results) <= rfc_acquired) * 100
                
                # Determine subjective luck message
                if percentile > 95:
                    luck_message = "You were supremely lucky! Congrats!"
                elif percentile >= 80:
                    luck_message = "You were very lucky!"
                elif percentile >= 60:
                    luck_message = "A bit above average luck."
                elif percentile >= 40:
                    luck_message = "Average luck."
                elif percentile >= 20:
                    luck_message = "A bit below average luck."
                elif percentile >= 5:
                    luck_message = "You were quite unlucky :("
                else:
                    luck_message = "You were supremely unlucky. Has a black cat crossed your path recently?"
                
                # Determine comparison message
                if percentile >= 50:
                    comparison = f"You were more lucky than {percentile:.1f}% of other players."
                else:
                    comparison = f"You were more unlucky than {100 - percentile:.1f}% of other players."
                
                # Calculate FC cost per rFC
                fc_per_rfc = total_fc / rfc_acquired if rfc_acquired > 0 else float('inf')
                
                # Display results
                st.write(f"**{luck_message}**")
                st.write(comparison)
                st.write(f"Your FC cost was {fc_per_rfc:.2f} FC spent per 1 rFC refined.")
