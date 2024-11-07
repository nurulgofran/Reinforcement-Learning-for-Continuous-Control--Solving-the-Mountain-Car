# mountain_car.py
import gym
import numpy as np
import matplotlib.pyplot as plt

# Initialize environment
env = gym.make('MountainCar-v0')

# Hyperparameters
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 2000
SHOW_EVERY = 500

# Discretize state space
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

# Initialize Q-table
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

def get_discrete_state(state):
    if isinstance(state, tuple):
        state = state[0]
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    # Clip values to be within valid range
    return tuple(np.clip(discrete_state.astype(np.int32), 0, 19))

# mountain_car.py - Training Loop Implementation
def main():
    episode_rewards = []
    
    for episode in range(EPISODES):
        episode_reward = 0
        state, _ = env.reset()  # Unpack the tuple
        discrete_state = get_discrete_state(state)
        done = False
        
        while not done:
            # Get action from Q-table
            action = np.argmax(q_table[discrete_state])
            
            # Take action in environment - handle 5 return values
            new_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated  # Combine both flags
            
            episode_reward += reward
            new_discrete_state = get_discrete_state(new_state)
            
            # Q-learning update
            if not done:
                max_future_q = np.max(q_table[new_discrete_state])
                current_q = q_table[discrete_state + (action,)]
                new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
                q_table[discrete_state + (action,)] = new_q
            else:
                q_table[discrete_state + (action,)] = reward  # Handle terminal state
            
            discrete_state = new_discrete_state
        
        episode_rewards.append(episode_reward)
        
        if episode % SHOW_EVERY == 0:
            print(f"Episode: {episode}, Reward: {episode_reward}")

    env.close()
    
    # Plot results
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Progress')
    plt.show()

if __name__ == "__main__":
    main()