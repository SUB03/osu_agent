import numpy as np
from collections import deque
from typing import NamedTuple
import uuid
import torch as th
import time

class Trajectory:
    def __init__(self, 
                 observation_space, 
                 action_space, 
                 device,
                 gamma: float = 0.99,
                 gae_lambda: float = 1,):
        self.id = uuid.uuid4() 
        self.observations = {}
        self.actions = {}
        self.rewards = {}
        self.dones = {}
        self.cumulative_reward = 0 
        self.values = {}
        self.advantages = {}
        self.log_probs = {}
        self.returns = {}
        self.current_step = 0  # Track the current step
        self.observation_space = observation_space
        self.action_space = action_space
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device

    def add_step(self,obs, action, reward, done, value, log_prob):

        # calculate the cumulative reward
        self.cumulative_reward += reward

        # Add the data to the respective dictionaries
        self.observations[self.current_step] = np.array(obs)
        self.actions[self.current_step] = np.array(action)
        self.rewards[self.current_step] = np.array(reward)
        self.values[self.current_step] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.current_step] = log_prob.clone().cpu().numpy()
        self.dones[self.current_step] = done
        

        # Prepare for the next step
        self.current_step += 1

    def get_data(self):
        # Assuming the steps are sequential and start from 0, and there are no gaps:
        num_steps = len(self.observations)
        # Create zero arrays for each type of data
        observations = np.zeros((num_steps,) + self.observation_space.shape , dtype=np.float32)

        actions = np.zeros((num_steps,) +  self.action_space.shape, dtype=np.float32)
        rewards = np.zeros(num_steps, dtype=np.float32)
        dones = np.zeros(num_steps, dtype=bool)
        advantages = np.zeros(num_steps, dtype=np.float32)
        returns = np.zeros(num_steps, dtype=np.float32)
        log_probs = np.zeros(num_steps, dtype=np.float32)
        values = np.zeros(num_steps, dtype=np.float32)
        

        # Fill the arrays
        for i in range(num_steps):
            observations[i] = self.observations[i]
            actions[i] = self.actions[i]
            rewards[i] = self.rewards[i]
            dones[i] = self.dones[i]
            advantages[i] = self.advantages[i]
            returns[i] = self.returns[i]
            log_probs[i] = self.log_probs[i]
            values[i] = self.values[i]
            
        return {
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "log_probs": log_probs,
            "advantages": advantages,
            "values": values,
            "returns": returns,
            "cumulative_reward": self.cumulative_reward,
            # Add "values", "advantages", "log_probs", "returns" similarly if needed
        }
        
    def compute_returns_and_advantage(self, best_trajectory, states_array, use_best_value=False, threshold=1):
        """
        Compute the Generalized Advantage Estimation (GAE) and returns for the trajectory.
        """
        steps = sorted(self.rewards.keys())
        horizon = len(steps)
        last_gae_lam = 0
                
        # Compute the advantages and returns
        for step in reversed(steps):
            next_non_terminal = 1.0 - self.dones[step]
            next_values = np.array(0.0, dtype=np.float32) if step == horizon - 1 else self.values[step + 1]
            
            # If the best trajectory is used, check if the current state is in the best trajectory
            if use_best_value:
                if best_trajectory.id == self.id:
                    current_value = self.values[step]
                else:
                    current_state = self.observations[step]
                    best_value, distance = self.search_state_in_trajectory(current_state, states_array, best_trajectory, return_distance= True)
                    
                    if best_value is not None and self.values[step] < best_value and distance <= threshold:
                        current_value = best_value
                    else:
                        current_value = self.values[step]
            else:
                current_value = self.values[step]

            # Compute the advantage
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - current_value
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam

        for key in self.advantages.keys():
            self.returns[key] = self.advantages[key] + self.values[key]


    def search_state_in_trajectory(self, current_state, states_array, trajectory, return_distance=False):
        """
        Searches for the closest state in the trajectory using NumPy for efficient computation.
        """
        if not isinstance(current_state, np.ndarray):
            current_state = np.array(current_state)

        distances = np.linalg.norm(states_array.squeeze() - current_state, axis=1)
        min_idx = np.argmin(distances)
        min_dist = distances[min_idx]
        best_value = trajectory.values[min_idx]

        return (best_value, min_dist) if return_distance else (best_value, min_dist == 0)


class TrajectoryBuffer:
    # trajectory buffer designed to store trajectories
    # initialize the buffer with a maximum size, observation space, action space, device, gamma, and gae_lambda
    
    def __init__(self, buffer_size, observation_space, action_space, device, gamma, gae_lambda):
        self.trajectories = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        self.current_trajectory = Trajectory(observation_space, action_space, device, gamma, gae_lambda)
        self.cache = []

    def add(self, obs, action, reward, done, value, log_prob):
        self.current_trajectory.add_step(obs, action, reward, done, value, log_prob)
        if done:
            self.trajectories.append(self.current_trajectory)
            self.current_trajectory = Trajectory(self.observation_space, self.action_space, self.device, self.gamma, self.gae_lambda)
            if len(self.trajectories) > self.buffer_size:
                self.trajectories.pop(0)
    
    def best_trajectory(self):
        # Find the trajectory with the highest cumulative reward
        max_cumulative_reward = -np.inf
        idx_of_max_reward = -1
        for idx, trajectory in enumerate(self.trajectories):
            if trajectory.cumulative_reward > max_cumulative_reward:
                max_cumulative_reward = trajectory.cumulative_reward
                idx_of_max_reward = idx
                
        best_trajectory = self.trajectories[idx_of_max_reward]
        return idx_of_max_reward, best_trajectory
            

        
                
    def sample_trajectories(self, batch_size_trajectory,):
        assert len(self.trajectories) >= batch_size_trajectory, "Not enough trajectories to sample"
        

        # Prepare indices for sampling, always including the trajectory with the highest reward
        indices_to_sample = list(range(len(self.trajectories)))
        # Remove the index of the trajectory with the highest cumulative reward to avoid duplicate
        
        idx_of_max_reward, _ = self.best_trajectory()
        indices_to_sample.remove(idx_of_max_reward)
        additional_samples_needed = batch_size_trajectory - 1  # One spot is reserved
        
        # Sample additional trajectories
        if additional_samples_needed > 0:
            sampled_indices = np.random.choice(indices_to_sample, additional_samples_needed, replace=False).tolist()
        else:
            sampled_indices = []
        
        # Add the trajectory with the highest reward
        sampled_indices.append(idx_of_max_reward)
        
        return sampled_indices
        
        
    
    def sample(self, sampled_trajectories_indices, buffer_size_sample, batch_size=None, use_best_value=False, threshold=0.01):
        num_sampled_trajectories = len(sampled_trajectories_indices)
        samples_per_trajectory = max(1, buffer_size_sample // num_sampled_trajectories)

        # Prepare lists to RolloutBufferSamples
        sampled_observations = []
        sampled_actions = []
        sampled_rewards = []
        sampled_values = []
        sampled_advantages = []
        sampled_log_probs = []
        sampled_dones = []
        sampled_returns = []
        
        # Step 1: Get the best trajectory if needed
        
        if use_best_value:
            best_trajectory_buffer = self.trajectories[sampled_trajectories_indices[-1]]
            states_array = np.array([state for state in best_trajectory_buffer.observations.values()])
            
        else:
            best_trajectory_buffer = None
            states_array = None
            
        # Step 2: Get samples from each trajectory
        for trajectory_index in sampled_trajectories_indices:
            num_observations = len(self.trajectories[trajectory_index].observations)
            num_samples = min(samples_per_trajectory, num_observations)
            
            
            # Sample indices
            if num_samples < num_observations:
                indices = np.random.choice(len(self.trajectories[trajectory_index].observations), num_samples, replace=False)
                            
            else:
                indices = np.arange(len(self.trajectories[trajectory_index].observations))  # Use all if not enough to sample

            # Compute returns and advantages if not already computed
            if self.trajectories[trajectory_index].id not in self.cache:
                self.trajectories[trajectory_index].compute_returns_and_advantage(best_trajectory_buffer, states_array, use_best_value=use_best_value, threshold=threshold)
                self.cache.append(self.trajectories[trajectory_index].id)
                
            # Add the samples to the lists
              
            for index in indices:
                sampled_observations.append(self.trajectories[trajectory_index].observations[index])
                sampled_actions.append(self.trajectories[trajectory_index].actions[index])
                sampled_rewards.append(self.trajectories[trajectory_index].rewards[index])
                sampled_dones.append(self.trajectories[trajectory_index].dones[index])
                sampled_values.append(self.trajectories[trajectory_index].values[index])
                sampled_advantages.append(self.trajectories[trajectory_index].advantages[index])
                sampled_log_probs.append(self.trajectories[trajectory_index].log_probs[index])
                sampled_returns.append(self.trajectories[trajectory_index].returns[index])

        # Convert lists to PyTorch tensors
        sampled_observations = self.standardize_shapes(sampled_observations)
        #sampled_actions = self.standardize_shapes(sampled_actions)
        sampled_observations = self.to_torch(np.array(sampled_observations), self.device).squeeze(1)
        sampled_actions = self.to_torch(np.array(sampled_actions), self.device)
        sampled_values = self.to_torch(np.array(sampled_values), self.device).squeeze(1)
        sampled_log_probs = self.to_torch(np.array(sampled_log_probs), self.device).squeeze(1)
        sampled_advantages = self.to_torch(np.array(sampled_advantages), self.device).squeeze(1)
        sampled_returns = self.to_torch(np.array(sampled_returns), self.device).squeeze(1)



        return self._batch_or_return_all(batch_size, sampled_observations, sampled_actions, sampled_values, sampled_log_probs, sampled_advantages, sampled_returns)

    # Yield the samples in batches if batch_size is provided, otherwise return all
    def _batch_or_return_all(self, batch_size, *data_tensors):
        if batch_size is None:
            return RolloutBufferSamples(*data_tensors)

        # Otherwise, yield minibatches
        total_samples = data_tensors[0].size(0)
        start_idx = 0
        while start_idx < total_samples:
            end_idx = min(start_idx + batch_size, total_samples)
            yield RolloutBufferSamples(*(tensor[start_idx:end_idx] for tensor in data_tensors))
            start_idx += batch_size
            
    # Standardize the shapes of the sampled data
    def standardize_shapes(self, sampled_list):

        standardized_list = []
        for arr in sampled_list:
            # Ensure the input is a numpy array
            if not isinstance(arr, np.ndarray):
                arr = np.array(arr)
            
            # Flatten if the array is higher than 1D, then reshape to 2D
            if arr.ndim > 1:
                arr = arr.flatten()
            elif arr.ndim == 0:  # Handle scalar case
                arr = np.expand_dims(arr, 0)

            # Reshape to 2D (makes 1D arrays into row vectors of shape [1, n])
            arr_2d = arr.reshape(1, -1)
            standardized_list.append(arr_2d)

        return standardized_list
    
    def to_torch(self, array: np.ndarray, device: th.device, copy: bool = True) -> th.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        """
        if copy:
            return th.tensor(array, device=device, dtype=th.float32)
        return th.as_tensor(array, device=device, dtype=th.float32)

# Named tuple to store the sampless
class RolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor