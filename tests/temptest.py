import numpy as np
import matplotlib.pyplot as plt

fam_exp_rate = 0.1
fam_age_size_range = np.arange(101) # size range e.g. range(0, 100 + 1)
fam_age_weights = np.exp(-fam_exp_rate * fam_age_size_range)
fam_age_weights /= np.sum(fam_age_weights) # normalize the weights so they sum to 1

plt.figure(figsize=(10, 6))  # Adjust figure size for better visibility
plt.bar(fam_age_size_range, fam_age_weights)  # Use a bar plot
plt.xlabel("Ages")
plt.ylabel("Normalized Weight")
plt.title("Exponential Decay Function applied to favour lower ages")

# Customizing x-axis ticks to show labels for every 10 years
plt.xticks(np.arange(0, 101, 10))

plt.show(block=False)

print("")
# cells_agents_timesteps = {}

# cells_agents_timesteps[1] = []

# cells_agents_timesteps[1].append([1000, 40, 143])

# population_per_timestep = [0 for its in range(144)]

# cell_agents_timesteps = cells_agents_timesteps[1]

# for agentid, starttimestep, endtimestep in cell_agents_timesteps:
#     for timestep in range(starttimestep, endtimestep+1):
#         population_per_timestep[timestep] += 1

# print(population_per_timestep)