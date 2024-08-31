import json
import pandas as pd
import matplotlib.pyplot as plt
import glob

# Function to load JSON data from files
def load_json_files(path):
    data = []
    for filename in glob.glob(path):
        with open(filename, 'r') as file:
            json_data = json.load(file)
            data.extend(json_data)
    return data

# Load the data
json_data_rewards = load_json_files('rewardsR1.json')
json_data_conflicts = load_json_files('conflictsR2.json')

# Convert to DataFrame
df_rewards = pd.DataFrame(json_data_rewards)
df_conflicts = pd.DataFrame(json_data_conflicts)

# Display the first few rows of the DataFrame
print(df_rewards.head())

# Extract data from DataFrame
episode = df_rewards.iloc[:, 1]
total_reward = df_rewards.iloc[:, 2]
conflicts = df_conflicts.iloc[:, 2]

# Set up a figure with two subplots
plt.figure(figsize=(10, 3))  # Wider figure to accommodate two subplots

# First subplot for Total Rewards
plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
plt.plot(episode, total_reward, 'b-')  # 'b-' is a blue solid line
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Rewards by Episode')
plt.grid(True)

# Second subplot for Conflicts
plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
plt.plot(episode, conflicts, 'r-')  # 'r-' is a red solid line
plt.xlabel('Episode')
plt.ylabel('Total Collisions')
plt.title('Total Collisions by Episode')
plt.grid(True)

# Adjust layout to not overlap subplots
plt.tight_layout()

# Show the plot
plt.show()
