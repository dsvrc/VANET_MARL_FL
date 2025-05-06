# vanet_env.py
import numpy as np
import gym
from gym import spaces

class VANETEnvironment(gym.Env):
    def __init__(self, data_processor, num_vehicles, num_rsus):
        super(VANETEnvironment, self).__init__()
        
        self.data_processor = data_processor
        self.num_vehicles = num_vehicles
        self.num_rsus = num_rsus
        self.current_timestamp = 0.0
        self.max_timestamp = 100.0 
        self.timestep = 0.1  
      
        self.action_space = spaces.Discrete(num_vehicles + num_rsus)
        
       
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(20,),  
            dtype=np.float32
        )
    
    def reset(self):
        self.current_timestamp = 0.0
        return self._get_observation()
    
    def step(self, action):
        reward = self._calculate_reward(action)
        
        self.current_timestamp += self.timestep
        if self.current_timestamp > self.max_timestamp:
            done = True
        else:
            done = False
            
        return self._get_observation(), reward, done, {}
    
    def _get_observation(self):
        vehicle_states = self.data_processor.get_vehicle_states(self.current_timestamp)
        comm_links = self.data_processor.get_communication_links(self.current_timestamp)
        
       
        observation = np.zeros(20)  
        
        if not vehicle_states.empty:
            veh = vehicle_states.iloc[0]
            observation[0:4] = [veh['x'], veh['y'], veh['speed'], veh['heading']]
            
        return observation
    
    def _calculate_reward(self, action):
       
        
        links = self.data_processor.get_communication_links(self.current_timestamp)
        
        action_vehicle_id = f"V{action:03d}"  
        
        link = links[links['target_id'] == action_vehicle_id]
        
        if link.empty:
            return -1.0 
        
        pdr = link['pdr'].values[0]
        latency = link['latency_ms'].values[0]
        snr = link['snr_db'].values[0]
        
        # Reward function prioritizing high PDR, low latency, high SNR
        reward = 2.0 * pdr - 0.01 * latency + 0.05 * snr
        
        return reward