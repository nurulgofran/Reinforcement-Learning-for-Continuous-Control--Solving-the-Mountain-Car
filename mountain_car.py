# mountain_car.py
import gym
import numpy as np
import matplotlib.pyplot as plt

# Initialize environment with rendering
env = gym.make("MountainCar-v0", render_mode="human")

# Hyperparameters
LEARNING_RATE = 0.1  # Adjust as needed
DISCOUNT = 0.99  # Increase to focus on future rewards
EPISODES = 10000  # Increase from 100 or 2000
SHOW_EVERY = 500

# Exploration parameters
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995  # Make decay slower

# Discretize state space
DISCRETE_OS_SIZE = [40, 40]  # Instead of [20, 20]
discrete_os_win_size = (
    env.observation_space.high - env.observation_space.low
) / DISCRETE_OS_SIZE

# Initialize Q-table
q_table = np.random.uniform(
    low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n])
)


def get_discrete_state(state):
    if isinstance(state, tuple):
        state = state[0]
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    # Clip values to be within valid range
    return tuple(np.clip(discrete_state.astype(np.int32), 0, 19))


# mountain_car.py - Training Loop Implementation
def main():
    episode_rewards = []
    epsilon = 1  # Exploration rate
    START_EPSILON_DECAYING = 1
    END_EPSILON_DECAYING = EPISODES // 2
    epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

    for episode in range(EPISODES):
        episode_reward = 0
        state, _ = env.reset()
        discrete_state = get_discrete_state(state)
        done = False

        while not done:
            if np.random.random() > epsilon:
                action = np.argmax(q_table[discrete_state])
            else:
                action = np.random.randint(0, env.action_space.n)

            new_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Update the reward calculation in your main loop
            if new_state[0] >= env.goal_position:
                reward = 0  # Goal reached
                terminated = True
            else:
                # Penalize each step to encourage faster learning
                reward = -1

            episode_reward += reward
            new_discrete_state = get_discrete_state(new_state)

            if not done:
                max_future_q = np.max(q_table[new_discrete_state])
                current_q = q_table[discrete_state + (action,)]
                new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (
                    reward + DISCOUNT * max_future_q
                )
                q_table[discrete_state + (action,)] = new_q
            else:
                q_table[discrete_state + (action,)] = reward  # No future state
                if terminated:
                    print(
                        f"Episode {episode} terminated at position: {new_state[0]:.2f}"
                    )
                if truncated:
                    print(f"Episode {episode} truncated (timeout)")

            discrete_state = new_discrete_state

        if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
            epsilon -= epsilon_decay_value

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        episode_rewards.append(episode_reward)
        if episode % SHOW_EVERY == 0:
            print(f"Episode: {episode}, Reward: {episode_reward}")

    env.close()
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Progress")
    plt.show()


if __name__ == "__main__":
    main()
