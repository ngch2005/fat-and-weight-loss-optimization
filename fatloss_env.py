import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class FatLossEnv(gym.Env):
    """
    Advanced RL environment for fat and weight loss optimization.
    Modified for Streamlit interactivity and Optuna HPO tracking.
    """

    def __init__(self):
        super(FatLossEnv, self).__init__()

        # Action space: 9 combined actions (3 Diet x 3 Activity)
        self.action_space = spaces.Discrete(9)

        self.action_names = {
            0: "1-Aggressive + Rest",
            1: "2-Aggressive + Light Activity",
            2: "3-Aggressive + High Exertion",
            3: "4-Moderate + Rest",
            4: "5-Moderate + Light Activity",
            5: "6-Moderate + High Exertion",
            6: "7-Maintenance + Rest",
            7: "8-Maintenance + Light Activity",
            8: "9-Maintenance + High Exertion"
        }

        # Observation space: 8 variables
        low = np.array([40.0, 4.0, 0.0, 0.0, 18.0, 140.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([200.0, 60.0, 12.0, 1.0, 80.0, 220.0, 365.0, 365.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.target_fat = None
        self.max_steps = 365

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)
        
        self.current_step = 0
        self.prev_action = None
        self.same_action_count = 0
        self.extreme_days = 0

        if options:
            self.gender = options.get("gender", random.choice([0, 1]))
            self.age = options.get("age", random.randint(18, 50))
            self.height = options.get("height", 170.0)
            self.true_weight = options.get("weight", 85.0)
            self.true_fat = options.get("fat", 30.0)
        else:
            self.gender = random.choice([0, 1])
            self.age = random.randint(18, 50)
            if self.gender == 1:
                self.height = random.uniform(165.0, 190.0)
                self.true_weight = random.uniform(80.0, 110.0)
            else:
                self.height = random.uniform(155.0, 175.0)
                self.true_weight = random.uniform(65.0, 95.0)
            self.true_fat = random.uniform(25.0, 50.0)

        height_in_meters = self.height / 100.0
        self.min_weight = 16 * (height_in_meters ** 2)
        self.target_weight = 22.0 * (height_in_meters ** 2) 
        
        # Biological Body Fat Calculations ---
        if self.gender == 1: 
            # Male Logic
            self.min_fat = 4.0      # Starvation level (punish)
            self.target_fat = 14.0  # Healthy athletic goal (reward)
        else: 
            # Female Logic
            self.min_fat = 12.0     # Starvation level (punish)
            self.target_fat = 22.0  # Healthy athletic goal (reward)

        # Older people naturally need slightly higher body fat. 
        # We can add a tiny modifier: +1% target fat for every 10 years over 30
        if self.age > 30:
            age_modifier = ((self.age - 30) // 10) * 1.0
            self.target_fat += age_modifier

        self.sleep_hours = 8.0

        self.state = np.array([
            self.true_weight, self.true_fat, self.sleep_hours, 
            self.gender, self.age, self.height,
            self.same_action_count, self.extreme_days
        ], dtype=np.float32)

        return self.state, self._get_info()

    def _calculate_bmr(self):
        base = 10 * self.true_weight + 6.25 * self.height - 5 * self.age
        return base + 5 if self.gender == 1 else base - 161

    def _get_info(self):
        """Centralized info dictionary for consistent tracking"""
        return {
            "True Weight": float(self.true_weight),
            "True Fat": float(self.true_fat),
            "Sleep": float(self.sleep_hours),
            "BMR": float(self._calculate_bmr()),
            "Step": self.current_step
        }

    def step(self, action):
        self.current_step += 1
        prev_weight = self.true_weight
        prev_fat = self.true_fat
        prev_sleep = self.sleep_hours

        # 1. Action Tracking
        if self.prev_action is not None and action == self.prev_action:
            self.same_action_count += 1
        else:
            self.same_action_count = 0
        self.prev_action = action

        diet_choice = action // 3
        activity_choice = action % 3

        if diet_choice == 0 or activity_choice == 2:
            self.extreme_days += 1
        else:
            self.extreme_days = max(0, self.extreme_days - 1)

        # 2. Physics & Energy Math
        bmr = self._calculate_bmr()
        activity_multiplier = [1.2, 1.375, 1.55][activity_choice]
        tdee = bmr * activity_multiplier
        calorie_intake = tdee - [800, 400, 0][diet_choice]
        net_energy = calorie_intake - tdee

        # 3. Penalties & Factors
        diet_efficiency = max(0.65, 1.0 - 0.05 * self.same_action_count) if diet_choice == 0 else 1.0
        sleep_factor = 0.7 if prev_sleep < 6.0 else (0.85 if prev_sleep < 7.0 else 1.0)

        # 4. Body Composition Update
        weight_change = net_energy / 7700.0 
        if net_energy < 0:
            fat_mass_change = (net_energy / 7700.0) * sleep_factor * diet_efficiency
        else:
            fat_mass_change = net_energy / 7700.0

        self.true_weight = max(self.min_weight, self.true_weight + weight_change)
        current_fat_mass = (prev_fat / 100.0) * prev_weight
        new_fat_mass = max(1.0, current_fat_mass + fat_mass_change)
        self.true_fat = (new_fat_mass / self.true_weight) * 100.0

        # 5. Sleep Dynamics
        if activity_choice == 2:
            self.sleep_hours -= 1.5 if diet_choice == 0 else 0.8
        elif activity_choice == 0:
            self.sleep_hours += 1.0 if diet_choice == 2 else 0.5
        elif activity_choice == 1 and diet_choice == 0:
            self.sleep_hours -= 0.3

        if self.extreme_days >= 3:
            self.sleep_hours -= 0.2 * (self.extreme_days - 2)
        self.sleep_hours = float(np.clip(self.sleep_hours, 0.0, 12.0))

        # 6. Reward Calculation
        R_fat = (prev_fat - self.true_fat) * 10.0
        if self.true_weight > self.target_weight:
            R_weight = (prev_weight - self.true_weight) * 2.0
        else:
            R_weight = 0.0
        R_sleep = -abs(8.0 - self.sleep_hours)
        
        act_score = [0, 1, 2][activity_choice]
        act_penalty = -15 if (activity_choice == 2 and (diet_choice == 0 or self.sleep_hours < 6.0)) else 0
        R_activity = act_score + act_penalty

        repeat_penalty = -0.2 * self.same_action_count if self.same_action_count >= 2 else 0.0
        extreme_penalty = -0.15 * self.extreme_days if self.extreme_days >= 3 else 0.0

        reward = (0.50 * R_fat) + (0.20 * R_weight) + (0.10 * R_sleep) + (0.05 * R_activity) 
        reward += repeat_penalty + extreme_penalty - 0.1

       # 7. Termination Logic
        terminated = False
        truncated = False

        fat_goal_met = self.true_fat <= self.target_fat
        weight_goal_met = self.true_weight <= self.target_weight

        # 1. FIRST: Safety Check (Health Crash = -100)
        if self.sleep_hours <= 3.0 or self.true_weight <= self.min_weight or self.true_fat <= self.min_fat:
            reward -= 100.0  
            terminated = True
            
        # 2. SECOND: Perfect Goal (Both met = +100 bonus)
        elif fat_goal_met and weight_goal_met:
            reward += 100.0  
            terminated = True
            
        # 3. THIRD: Standard Goal (Either Fat OR Weight met = +50 reward)
        # This is the change you requested
        elif fat_goal_met or weight_goal_met:
            reward += 50.0   
            terminated = True

        # 4. FOURTH: Time Out (End after 1 year)
        elif self.current_step >= self.max_steps:
            truncated = True

        # 8. Update State
        self.state = np.array([
            self.true_weight, self.true_fat, self.sleep_hours, 
            self.gender, self.age, self.height, 
            self.same_action_count, self.extreme_days
        ], dtype=np.float32)

        info = self._get_info()
        info.update({"R_fat": R_fat, "R_sleep": R_sleep, "R_activity": R_activity})
        
        return self.state, float(reward), terminated, truncated, info