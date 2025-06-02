import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

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

# Function to calculate strategy metrics for summary table and plan tab
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

# ********** Streamlit app UI **********
# Sidebar menu for calculator selection
calculator = st.sidebar.selectbox("WOS Calculators", ["FC Refinement"])

if calculator == "FC Refinement":
    # Horizontal tabs with reordered tabs
    tab1, tab2, tab3, tab4 = st.tabs(["rFC Simulator", "Was I Lucky?", "Plan", "Strategies"])
    
    with tab1: # Run simulations tab
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
        simulations = st.number_input("Number of Monte Carlo Simulations (1,000=fast; 4,000=balanced; 100,000=precise)", min_value=1, max_value=100000, value=4000)
        
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
    
    with tab2: # Was I Lcuky? tab
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
    
    with tab3: # Plan Tab
        st.subheader("Plan")
        st.write("Determine weekly refinements needed to reach a specified rFC target in a given period of time.")
        if not 'rfc_needed' in st.session_state:
            st.session_state.rfc_needed = 0
        
        # Input UI elements
        opt1 = "Choose buildings"
        opt2 = "Enter rFC goal"
        plan_input_method = st.segmented_control("rFC Target specification method",[opt1, opt2],
                                                 selection_mode="single",label_visibility="hidden",default=opt1)
        if plan_input_method == opt1:
            # Load the JSON file
            with open("building_costs.json", "r") as f:
                data = json.load(f)

            # List of levels and buildings for easy reference
            LEVELS = ["FC6", "FC7", "FC8", "FC9", "FC10"]
            BUILDINGS = list(data["buildings"].keys())
            SHORT_NAMES = {
                "Furnace": "Furn.",
                "Embassy": "Emb.",
                "Infirmary": "Infirm.",
                "Command Center": "C.C.",
                "Infantry Camp": "C-I",
                "Marksman Camp": "C-M",
                "Lancer Camp": "C-L",
                "War Academy": "W.A."
            }
            # Initialize session state for selected levels and buildings
            if "selected_levels" not in st.session_state:
                st.session_state.selected_levels = []
            if "selected_buildings" not in st.session_state:
                st.session_state.selected_buildings = []

            # Pills widget for level selection (FC6 to FC10)
            #st.markdown("**Select building upgrades to determine rFC target**")
            selected_levels = st.pills(
                "Building Level(s) (multiselect)",
                options=LEVELS,
                selection_mode="multi",
                key="level_pills"
            )

            # Update session state with selected levels
            st.session_state.selected_levels = selected_levels if selected_levels else []

            # For each selected level, display a row of pills for buildings
            if st.session_state.selected_levels:
                for level in st.session_state.selected_levels:
                    # Create a list of building labels with shortened names prefixed by level
                    building_labels = [f"{level.lower()}: {SHORT_NAMES[building]}" for building in BUILDINGS]
                    building_options = list(zip(building_labels, [(level, building) for building in BUILDINGS]))
                    
                    # Pills widget for buildings with multi-selection
                    selected_building_labels = st.pills(
                        f"Buildings for {level}",
                        options=[label for label, _ in building_options],
                        selection_mode="multi",
                        key=f"buildings_pills_{level}",
                        label_visibility="hidden"
                    )
                    
                    # Update selected buildings for this level
                    # First, remove any previous selections for this level
                    st.session_state.selected_buildings = [
                        (sel_level, sel_building) for sel_level, sel_building in st.session_state.selected_buildings
                        if sel_level != level
                    ]
                    # Add the new selections for this level
                    if selected_building_labels:
                        new_selections = [
                            opt[1] for opt in building_options if opt[0] in selected_building_labels
                        ]
                        st.session_state.selected_buildings.extend(new_selections)

            # Filter selected buildings to only include those from currently selected levels
            st.session_state.selected_buildings = [
                (level, building) for level, building in st.session_state.selected_buildings
                if level in st.session_state.selected_levels
            ]

            # Display the currently selected buildings
            if st.session_state.selected_buildings:
                st.write("Upgrade goal:", ", ".join(
                    f"{level.lower()}: {building}" for level, building in st.session_state.selected_buildings
                ))

            total_fc = 0
            total_rfc = 0

            if st.session_state.selected_buildings:
                for level, building in st.session_state.selected_buildings:
                    total = data["buildings"][building][level]["total"]
                    total_fc += total["FC"] # Accumulate totals
                    total_rfc += total["rFC"] # Accumulate totals
                st.session_state.rfc_needed = total_rfc
                st.session_state.fc_needed_build = total_fc
                st.write(f"To upgrade selected buildings, you need {total_rfc} rFC and {total_fc} FC")

        elif plan_input_method == opt2:
            st.session_state.rfc_needed = st.number_input("Enter number of rFC you need", min_value=1, max_value=1000, value=1, format="%d")

        current_rfc = st.number_input("Enter number of rFC you already have", min_value=0, max_value=1000, value=0, format="%d")
        weeks = st.number_input(f"Enter weeks you want to reach rFC target", min_value=1, max_value=104, value=1, format="%d")
        luck_option = st.selectbox("Accounting for luck, how sure do you want to be to reach target on schedule?", 
                                  ["Assume average luck", "Plan for bad luck", "I'm feeling lucky!", "Custom"])
        
        # Set percentile based on luck option
        if luck_option == "Assume average luck":
            percentile = 50.0
            editable = False
        elif luck_option == "Plan for bad luck":
            percentile = 10.0
            editable = False
        elif luck_option == "I'm feeling lucky!":
            percentile = 90.0
            editable = False
        else:  # custom
            percentile = 50.0
            editable = True
        
        # Percentile input, editable only for custom
        percentile_factor = st.number_input("Luck percentile", min_value=0.0, max_value=100.0, value=percentile, 
                                           step=0.1, disabled=not editable, format="%.1f")
        
        # Create plan button
        if st.button("Create plan"):
            if st.session_state.rfc_needed == 0:
                st.warning("Select buildings or number of rFC you need first!")
            else:
                # Calculate rFC needed to gain
                rfc_to_gain = st.session_state.rfc_needed - current_rfc
                if rfc_to_gain <= 0:
                    st.error("You already have enough rFC to meet your target!")
                else:
                    # Initialize variables
                    optimal_x = None
                    target_percentile_rfc = None
                    total_fc_used = None
                    
                    if luck_option == "Assume average luck":
                        # Step 1: Use theoretical expected rFC for 50th percentile
                        low, high = 1, 94  # Max 94 to allow 1 per day for days 2-7 (94 + 6 = 100)
                        while low <= high:
                            mid = (low + high) // 2
                            day1_refinements = mid
                            days2_7_refinements = 1
                            total_fc, _, expected_rfc = calculate_strategy_metrics(day1_refinements, days2_7_refinements)
                            total_expected_rfc = expected_rfc * weeks
                            
                            if total_expected_rfc >= rfc_to_gain:
                                optimal_x = mid
                                target_percentile_rfc = total_expected_rfc
                                total_fc_used = total_fc * weeks
                                high = mid - 1  # Try for a lower X
                            else:
                                low = mid + 1  # Need a higher X
                    else:
                        # Step 1: Set bounds using theoretical min_rfc and max_rfc
                        target_rfc_per_week = rfc_to_gain / weeks
                        low, high = 1, 94
                        min_x, max_x = None, None
                        
                        # Find minimum X where min_rfc * weeks >= rfc_to_gain
                        l, h = 1, 94
                        while l <= h:
                            mid = (l + h) // 2
                            day1_refinements = mid
                            days2_7_refinements = 1
                            _, min_rfc, _ = calculate_strategy_metrics(day1_refinements, days2_7_refinements)
                            total_min_rfc = min_rfc * weeks
                            
                            if total_min_rfc >= rfc_to_gain:
                                min_x = mid
                                h = mid - 1
                            else:
                                l = mid + 1
                        
                        # Find minimum X where max_rfc * weeks >= rfc_to_gain
                        l, h = 1, 94
                        while l <= h:
                            mid = (l + h) // 2
                            day1_refinements = mid
                            days2_7_refinements = 1
                            _, _, expected_rfc = calculate_strategy_metrics(day1_refinements, days2_7_refinements)
                            total_max_rfc = calculate_max_rfc(day1_refinements + 6) * weeks
                            
                            if total_max_rfc >= rfc_to_gain:
                                max_x = mid
                                h = mid - 1
                            else:
                                l = mid + 1
                        
                        # Set bounds for binary search
                        high = min_x if min_x is not None else 94
                        low = max_x if max_x is not None else 1
                        
                        # st.write(f"low={low}") # for debugging
                        # st.write(f"high={high}")

                        # Step 2: Approximate using theoretical expected rFC within bounds
                        initial_x = None
                        while low <= high:
                            mid = (low + high) // 2
                            day1_refinements = mid
                            days2_7_refinements = 1
                            _, _, expected_rfc = calculate_strategy_metrics(day1_refinements, days2_7_refinements)
                            total_expected_rfc = expected_rfc * weeks
                            
                            if total_expected_rfc >= rfc_to_gain:
                                initial_x = mid
                                high = mid - 1  # Try for a lower X
                            else:
                                low = mid + 1  # Need a higher X
                        
                        # Step 3: Refine with Monte Carlo simulations
                        simulations = 1000
                        
                        if initial_x is not None:
                            # Test a range around initial_x
                            for x in range(max(1, initial_x - 10), min(95, initial_x + 11)):
                                day1_refinements = x
                                days2_7_refinements = 1
                                rfc_results = []
                                for _ in range(simulations):
                                    fc, rfc = run_simulation(weeks, day1_refinements, days2_7_refinements)
                                    rfc_results.append(rfc)
                                percentile_rfc = np.percentile(rfc_results, percentile_factor)
                                
                                if percentile_rfc >= rfc_to_gain and (optimal_x is None or x < optimal_x):
                                    optimal_x = x
                                    target_percentile_rfc = percentile_rfc
                                    total_fc_used = fc
                        
                        # Step 4: Fallback to binary search within bounds if no solution found
                        if optimal_x is None:
                            low = min_x if min_x is not None else 1
                            high = max_x if max_x is not None else 94
                            while low <= high:
                                mid = (low + high) // 2
                                day1_refinements = mid
                                days2_7_refinements = 1
                                rfc_results = []
                                for _ in range(simulations):
                                    fc, rfc = run_simulation(weeks, day1_refinements, days2_7_refinements)
                                    rfc_results.append(rfc)
                                percentile_rfc = np.percentile(rfc_results, percentile_factor)
                                
                                if percentile_rfc >= rfc_to_gain:
                                    optimal_x = mid
                                    target_percentile_rfc = percentile_rfc
                                    total_fc_used = fc
                                    high = mid - 1  # Try for a lower X
                                else:
                                    low = mid + 1  # Need a higher X
                    
                    if optimal_x is None:
                        st.error("No strategy can achieve the target rFC within the given time and luck constraints :(")
                    else:
                        # Output formatted result
                        total_rfc = current_rfc + target_percentile_rfc
                        probability = percentile_factor
                        st.write(f"Using the weekly refinement strategy {optimal_x}/1/1/1/1/1/1, you have a {probability:.1f}% chance of obtaining {target_percentile_rfc:.0f} rFC in {weeks} weeks. Combined with your existing amount, you should end up with {total_rfc:.0f} rFC at a cost of {total_fc_used:.0f} FC expended in refinement.")
                        if plan_input_method == opt1:
                            fc_needed_combined = st.session_state.fc_needed_build + total_fc_used
                            fc_combined_per_week = fc_needed_combined/weeks
                            st.write(f"In addition, you will need {st.session_state.fc_needed_build} FC more for the selected building upgrades.")
                            st.write(f"FC needed for building and refinement, {st.session_state.fc_needed_build:.0f} + {total_fc_used:.0f} = {fc_needed_combined:.0f}, which is an average accumulation requirement of {fc_combined_per_week:.0f} per week (ignoring any FC which are currently in the backpack).")
                            
                            # Display the per-tier cost of each selected building in tables
                            st.divider()
                            st.write("#### Building Details")
                            if st.session_state.selected_buildings:
                                fc_data = []
                                rfc_data = []
                                for level, building in st.session_state.selected_buildings:
                                    stages = data["buildings"][building][level]["stages"]
                                    total = data["buildings"][building][level]["total"]
                                    
                                    # FC row
                                    fc_row = {
                                        "Level": level,
                                        "Building": building,
                                        "Total FC": total["FC"],
                                        "Stage1": stages[0]["FC"],
                                        "Stage2": stages[1]["FC"],
                                        "Stage3": stages[2]["FC"],
                                        "Stage4": stages[3]["FC"],
                                        "Stage5": stages[4]["FC"]
                                    }
                                    fc_data.append(fc_row)
                                    
                                    # rFC row
                                    rfc_row = {
                                        "Level": level,
                                        "Building": building,
                                        "Total rFC": total["rFC"],
                                        "Stage1": stages[0]["rFC"],
                                        "Stage2": stages[1]["rFC"],
                                        "Stage3": stages[2]["rFC"],
                                        "Stage4": stages[3]["rFC"],
                                        "Stage5": stages[4]["rFC"]
                                    }
                                    rfc_data.append(rfc_row)
                                
                                # Display FC table
                                st.markdown("**FC Cost Breakdown**")
                                st.dataframe(
                                    pd.DataFrame(fc_data),
                                    use_container_width=True,
                                    hide_index=True
                                )
                                
                                # Display rFC table
                                st.markdown("**rFC Cost Breakdown**")
                                st.dataframe(
                                    pd.DataFrame(rfc_data),
                                    use_container_width=True,
                                    hide_index=True
                                )
    
    with tab4:
        st.subheader("Summary of Strategies")
        st.write("In most situations, it's recommended to select a number of refinements the first day of the week and then refine 1x over the remaining 6 days. The following table summarizes a few reasonable strategies in order of increasing speed, but decreasing efficiency.")
        
        # Define list of common strategies
        strategies = [
            {"description": "f2p/low spender", "day1_refinements": 1, "days2_7_refinements": 1},
            {"description": "f2p/low spender", "day1_refinements": 14, "days2_7_refinements": 1},
            {"description": "Low/mid spender", "day1_refinements": 20, "days2_7_refinements": 1},
            {"description": "Mid/high spender", "day1_refinements": 40, "days2_7_refinements": 1},
            {"description": "Whale", "day1_refinements": 60, "days2_7_refinements": 1},
            {"description": "Whale", "day1_refinements": 80, "days2_7_refinements": 1},
            {"description": "Nobody", "day1_refinements": 94, "days2_7_refinements": 1}
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
                "Recommended for": strategy["description"],
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
            'Ave rFC/week': '{:.2f}',
            'Ave FC/rFC': '{:.2f}'
        })
        
        # Display the table
        st.dataframe(styled_strategy_df, use_container_width=True)
        
        # Plot FC/rFC ratio and average rFC as a function of weekly refinements
        st.write("Plot showing how the FC/rFC cost and expected rFC output vary with number of weekly refinements, assuming (total refinements - 6) are done on day one and one refinement each on the remaining six days. Lower FC/rFC ratios indicate better efficiency, while higher rFC values represent faster progress.")
        
        max_refinements = st.slider("Adjust upper plot limit", min_value=10, max_value=100, value=100, step=1)

        weekly_refinements = np.arange(7, max_refinements + 1, 1)
        fc_rfc_ratios = []
        avg_rfcs = []
        
        for total_refinements in weekly_refinements:
            day1_refinements = total_refinements - 6
            days2_7_refinements = 1
            total_fc, _, expected_rfc = calculate_strategy_metrics(day1_refinements, days2_7_refinements)
            fc_rfc_ratio = total_fc / expected_rfc if expected_rfc > 0 else float('inf')
            fc_rfc_ratios.append(fc_rfc_ratio)
            avg_rfcs.append(expected_rfc)
        
        fig, ax1 = plt.subplots()
        
        # Plot FC/rFC ratio on left y-axis
        ax1.plot(weekly_refinements, fc_rfc_ratios, 'b-', label='FC/rFC Cost Ratio')
        ax1.set_xlabel('Refinements per Week')
        ax1.set_ylabel('FC/rFC Cost Ratio', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        
        # Add major and minor gridlines
        ax1.grid(True, which='major', linestyle='-', linewidth=0.8)
        ax1.minorticks_on()
        ax1.grid(True, which='minor', linestyle='--', linewidth=0.4)
        
        # Create right y-axis for average rFC
        ax2 = ax1.twinx()
        ax2.plot(weekly_refinements, avg_rfcs, 'r-', label='Typical rFC/week')
        ax2.set_ylabel('Typical rFC made per week', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        plt.title('Refinement Speed-Cost Comparison')
        fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
        
        # Display plot
        st.pyplot(fig)
