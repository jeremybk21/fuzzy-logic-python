### Fuzzy Logic Cart Pole Control
### Description: This script uses fuzzy logic to control a cart pole system. The cart pole system is simulated using the OpenAI Gymnasium CartPole environment.
### Author: Jeremy B. Kimball
### Date: 2022-03-04

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from cartpole_continuous_action import ContinuousActionCartPoleEnv
from fuzzylogic import plot_fuzzy_sets, ZShapeFuzzySet, FuzzyRule, FuzzySystem, ConstantFuzzySet

# Define fuzzy sets

theta_set_neg = ZShapeFuzzySet(-0.01, 0.01, name="theta_neg", input_num=0)
theta_set_pos = ZShapeFuzzySet(-0.01, 0.01, inverted=True, name="theta_pos", input_num=0)
plot_fuzzy_sets([theta_set_neg, theta_set_pos], x_min=-0.5, x_max=0.5, title="Angle fuzzy set (radians)")

theta_dot_set_neg = ZShapeFuzzySet(-0.5, 0.5, name="theta_dot_neg", input_num=1)
theta_dot_set_pos = ZShapeFuzzySet(-0.5, 0.5, inverted=True, name="theta_dot_pos", input_num=1)
plot_fuzzy_sets([theta_dot_set_neg, theta_dot_set_pos], -10, 10, title="Angular velocity fuzzy set (radians/s)")

position_setpoint = 0 # Set the cart position setpoint desired location to the center of the track
cart_pos_set_neg = ZShapeFuzzySet(-0.3+position_setpoint, 0.3+position_setpoint, name="cart_position_neg", input_num=2)
cart_pos_set_pos = ZShapeFuzzySet(-0.3+position_setpoint, 0.3+position_setpoint, inverted=True, name="cart_position_pos", input_num=2)
plot_fuzzy_sets([cart_pos_set_neg, cart_pos_set_pos], -2, 2, title="Cart position fuzzy set (meters)")

cart_vel_set_neg = ZShapeFuzzySet(-0.5, 0.5, name="cart_velocity_neg", input_num=3)
cart_vel_set_pos = ZShapeFuzzySet(-0.5, 0.5, inverted=True, name="cart_velocity_pos", input_num=3)
plot_fuzzy_sets([cart_vel_set_neg, cart_vel_set_pos], -2, 2, title="Cart velocity fuzzy set (meters/s)")

force_set_med_left = ConstantFuzzySet(-1.25, name="force_med_left")
force_set_med_right = ConstantFuzzySet(1.25, name="force_med_right")
force_set_large_left = ConstantFuzzySet(-2.5, name="force_large_left")
force_set_large_right = ConstantFuzzySet(2.5, name="force_large_right")
force_set_small_left = ConstantFuzzySet(-0.6, name="force_small_left")
force_set_small_right = ConstantFuzzySet(0.6, name="force_small_right")
plot_fuzzy_sets([force_set_med_left, force_set_med_right, force_set_large_left, force_set_large_right, force_set_small_left, force_set_small_right], -4, 4, 
                title="Force fuzzy set (N)", n_points=1000, POI = [-1.25, 1.25, -2.5, 2.5, -0.6, 0.6])

# Define fuzzy rules

theta_rules = [
    FuzzyRule(theta_set_neg & theta_dot_set_neg, force_set_large_left), # If theta is negative and theta_dot is negative, apply a large force to the left
    FuzzyRule(theta_set_pos & theta_dot_set_neg, force_set_med_left), # If theta is positive and theta_dot is negative, apply a medium force to the left
]

theta_dot_rules = [
    FuzzyRule(theta_set_pos & theta_dot_set_pos, force_set_large_right), # If theta is positive and theta_dot is positive, apply a large force to the right
    FuzzyRule(theta_set_neg & theta_dot_set_pos, force_set_med_right), # If theta is negative and theta_dot is positive, apply a medium force to the right
]

cart_pos_rules = [
    FuzzyRule(cart_pos_set_neg, force_set_small_left), # If cart_pos is negative, apply a small force to the left
    FuzzyRule(cart_pos_set_pos, force_set_small_right), # If cart_pos is positive, apply a small force to the right
]

cart_vel_rules = [
    FuzzyRule(cart_vel_set_neg, force_set_med_left), # If cart_vel is negative, apply a medium force to the left
    FuzzyRule(cart_vel_set_pos, force_set_med_right), # If cart_vel is positive, apply a medium force to the right
]

rules = theta_rules + theta_dot_rules + cart_pos_rules + cart_vel_rules

# Use rules to define fuzzy system
system = FuzzySystem(rules)

# Use fuzzy system to control cart pole

env = ContinuousActionCartPoleEnv(render_mode="human")
observation = env.reset(seed = 9)

F = np.float32(0)

applied_forces = []
angles = []
angular_velocities = []
positions = []
positions_setpoint = []
times = []

for i in range(6000):

  observation, reward, terminated, truncated, info = env.step(F)

  # Setting position setpoints for different times in the simulation

  if i < 1800:
    cart_position = observation[0] - 1
    positions_setpoint.append(1)
    positions.append(cart_position + 1)
  elif i < 3600:
    cart_position = observation[0] + 1
    positions_setpoint.append(-1)
    positions.append(cart_position - 1)
  else:
    cart_position = observation[0]
    positions_setpoint.append(0)
    positions.append(cart_position)

  velocity = observation[1]
  angle = observation[2]
  angular_velocity = observation[3]

  F = system.output([angle, angular_velocity, cart_position, velocity])

  print("angle: ", angle)
  print("angular_velocity: ", angular_velocity)
  print("cart_position: ", cart_position)
  print("velocity: ", velocity)
  print("F: ", F)

  F = np.float32(F)

  if F > 0:
    F = np.float32(1)
  elif F <= 0:
    F = np.float32(-1)

  applied_forces.append(F)
  angles.append(angle)
  angular_velocities.append(angular_velocity)
  times.append(i*0.02)

  if terminated or truncated:
    observation = env.reset()
env.close()

# Plot results
fig, ax = plt.subplots(4, 1, figsize=(10, 10))
ax[0].plot(times, angles)
ax[0].set_title("Angle")
ax[0].set_ylabel("Angle (rad)")
ax[1].plot(times, angular_velocities)
ax[1].set_title("Angular velocity")
ax[1].set_ylabel("Angular velocity (rad/s)")
ax[2].plot(times, positions)
ax[2].plot(times, positions_setpoint)
ax[2].set_title("Cart position")
ax[2].set_ylabel("Position (m)")
ax[2].legend(["Cart position", "Setpoint"])
ax[3].plot(times, applied_forces)
ax[3].set_title("Applied force")
ax[3].set_xlabel("Time (s)")
ax[3].set_ylabel("Force (N)")
plt.tight_layout()
plt.show()