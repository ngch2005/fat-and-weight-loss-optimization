import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, DQN
from fatloss_env import FatLossEnv

# --- Page Config ---
st.set_page_config(page_title="Fat and Weight Loss Optimization Dashboard", layout="wide")

st.title("🏃‍♂️ Fat and Weight Loss Optimization")
st.markdown("""
This application showcases a Reinforcement Learning agent trained to optimize fat loss 
while maintaining healthy sleep patterns and preventing physical burnout.
""")

# --- Sidebar: User Inputs ---
st.sidebar.header("📋 Virtual Client Profile")

gender = st.sidebar.selectbox("Gender", options=["Male", "Female"], index=0)
gender_val = 1 if gender == "Male" else 0

age = st.sidebar.slider("Age", 18, 100, 25)
height = st.sidebar.number_input("Height (cm)", min_value=120.0, max_value=220.0, value=160.0)
weight = st.sidebar.number_input("Starting Weight (kg)", min_value=30.0, max_value=200.0, value=85.0)
fat_pct = st.sidebar.number_input("Starting Fat %", min_value=25.0, max_value=50.0, value=30.0)

# --- Calculate and Display Targets Before Running ---
height_in_meters = height / 100.0
calc_min_weight = 18.5 * (height_in_meters ** 2)

# Set base target fat percentage
if gender_val == 1:  # Male
    calc_target_fat = 14.0
else:  # Female
    calc_target_fat = 22.0

# Age modifier
if age > 30:
    age_modifier = ((age - 30) // 10) * 1.0
    calc_target_fat += age_modifier

# 2. Calculate Ideal Body Fat in kg using the image formula: Total Weight * Body Fat Percentage
target_fat_mass_kg = weight * (calc_target_fat / 100.0) 

# st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Optimization Goals")
st.sidebar.info(f"**Target Fat %:** {calc_target_fat:.1f}%\n\n**Target Fat Mass:** {target_fat_mass_kg:.1f} kg\n\n**Safe Minimum Weight:** {calc_min_weight:.1f} kg")

st.sidebar.markdown("---")
st.sidebar.header("🤖 Model Selection")
model_type = st.sidebar.radio("Select Trained Agent", ["PPO", "DQN"])

# --- Helper Function: Run Simulation ---
def run_animated_simulation(selected_model, user_options):
    env = FatLossEnv() 
    
    # Load the model with safety block
    try:
        if selected_model == "PPO":
            model = PPO.load("ppo_model")
        else:
            model = DQN.load("dqn_model")
    except Exception as e:
        st.error(f"Failed to load {selected_model} model. Make sure it is trained and saved in this folder! Error: {e}")
        st.stop()
    
    obs, info = env.reset(options=user_options)
    
    # Create empty placeholders for the live elements
    st.subheader(f"🚀 Live {selected_model} Simulation")
    chart_col, metrics_col = st.columns([2, 1])
    
    with chart_col:
        weight_chart = st.empty() 
    
    with metrics_col:
        st.write("### ⚡ Live Status")
        day_text = st.empty()
        weight_text = st.empty()
        fat_text = st.empty()
        sleep_text = st.empty()
        reward_text = st.empty() 

    history = []
    terminated = truncated = False
    
    # Run the loop with animation
    while not terminated and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        history.append({
            "Day": info["Step"],
            "Weight": info["True Weight"],
            "Fat_Pct": info["True Fat"],   
            "Sleep": info["Sleep"],
            "Action": env.action_names[int(action)],
            "Reward": float(reward) 
        })
        
        # Update text metrics
        day_text.write(f"📅 **Day:** {info['Step']}")
        weight_text.write(f"⚖️ **Weight:** {info['True Weight']:.1f} kg")
        fat_text.write(f"🔥 **Fat:** {info['True Fat']:.1f} %")
        sleep_text.write(f"😴 **Sleep:** {info['Sleep']:.1f} hrs")
        reward_text.write(f"⭐ **Reward:** {reward:.2f}") # Live reward update
        
      
        # Create the dataframe
        df_live = pd.DataFrame(history)
        
        # Rename the columns so the chart legend looks clean and formatted
        chart_data = df_live.rename(columns={"Weight": "Weight (KG)", "Fat_Pct": "Fat (%)"})
        
        # Plot using the new formatted names
        weight_chart.line_chart(chart_data.set_index("Day")[["Weight (KG)", "Fat (%)"]], x_label="Day")
        
        time.sleep(0.01) 
        
    return pd.DataFrame(history), info

# --- Execution ---
if st.sidebar.button("🚀 Run"):
    user_options = {
        "gender": gender_val,
        "age": age,
        "height": height,
        "weight": weight,
        "fat": fat_pct
    }
    
    # Run the animated version
    df_results, final_info = run_animated_simulation(model_type, user_options)
    total_reward = df_results['Reward'].sum()
    
    st.success(f"Simulation Complete! The Total Reward is: {total_reward:.2f}")
    
    # --- Metrics Section ---
    st.subheader("📊 Key Performance Metrics")
    m1, m2, m3, m4, m5 = st.columns(5)
    
    weight_diff = final_info['True Weight'] - weight
    fat_diff = final_info['True Fat'] - fat_pct
    
    m1.metric("Final Weight", f"{final_info['True Weight']:.1f} kg", f"{weight_diff:.1f} kg")
    m2.metric("Final Fat %", f"{final_info['True Fat']:.1f} %", f"{fat_diff:.1f} %")
    m3.metric("Avg. Sleep", f"{df_results['Sleep'].mean():.1f} hrs")
    m4.metric("Days Active", f"{len(df_results)}")
    m5.metric("Total Reward", f"{total_reward:.1f}")

    # --- Charts Section ---
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.write("### 📉 Weight & Fat Loss Trajectory")
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        
        ax1.plot(df_results['Day'], df_results['Weight'], color='blue', label='Weight (kg)')
        ax2.plot(df_results['Day'], df_results['Fat_Pct'], color='orange', label='Fat %')
        
        ax1.set_xlabel('Day')
        ax1.set_ylabel('Weight (kg)', color='blue')
        ax2.set_ylabel('Body Fat %', color='orange')
        st.pyplot(fig)

    with col_b:
        st.write("### 😴 Sleep & Recovery Monitor")
        st.line_chart(df_results.set_index('Day')['Sleep'], 
                      x_label="Day",
                      y_label="Sleep (Hours)")
        st.info("The model tries to keep sleep above 6-7 hours to avoid the 'Extreme Days' penalty.")

    # 4. --- Final Reward Graph ---
    st.markdown("---")
    st.write("### ⭐ Final Reward Graph")
    st.line_chart(
        df_results.set_index('Day')['Reward'], 
        x_label="Day", 
        y_label="Reward Score"
    )

    # --- Action Breakdown ---
    st.write("### 📑 Daily Prescription Log")
    st.dataframe(df_results, use_container_width=True)
    
    # --- Action Distribution Pie Chart ---
    st.write("### 🧩 Strategic Preference")
    action_counts = df_results['Action'].value_counts()
    fig_pie, ax_pie = plt.subplots(figsize=(4, 4))
    ax_pie.pie(action_counts, labels=action_counts.index, autopct='%1.1f%%', startangle=90)
    ax_pie.axis('equal') 
    st.pyplot(fig_pie, use_container_width=False)

else:
    st.info("Adjust the client profile on the left and click 'Run' to see the simulation.")