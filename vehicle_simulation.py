# vehicle_simulation.py
import os
import numpy as np
import pandas as pd
import time
import requests
import json
from data_processor import VANETDataProcessor
from vanet_env import VANETEnvironment
from rl_agent import DQNAgent

VEHICLE_ID = int(os.environ.get('VEHICLE_ID', 0))
NUM_VEHICLES = int(os.environ.get('NUM_VEHICLES', 5))
NUM_RSUS = int(os.environ.get('NUM_RSUS', 2))
AGGREGATION_SERVER = os.environ.get('AGGREGATION_SERVER', 'http://localhost:5000')

data_processor = VANETDataProcessor()
data_processor.load_data('vehicle_data.csv', 'communication_data.csv')

env = VANETEnvironment(data_processor, NUM_VEHICLES, NUM_RSUS)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

agent = DQNAgent(state_size, action_size)

EPISODES = 100
BATCH_SIZE = 32
FED_UPDATE_FREQUENCY = 10 

def train_local_model():
    """Train the local model for one episode"""
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0
    
    for time_step in range(500): 
        action = agent.act(state)
        
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        
        agent.remember(state, action, reward, next_state, done)
        
        state = next_state
        total_reward += reward
        
        if done:
            break
            
    if len(agent.memory) > BATCH_SIZE:
        agent.replay(BATCH_SIZE)
        
    return total_reward

def send_weights_to_server():
    weights = agent.get_weights()
    weights_data = {
        'vehicle_id': VEHICLE_ID,
        'weights': [w.tolist() for w in weights]
    }
    
    try:
        response = requests.post(
            f"{AGGREGATION_SERVER}/update_weights",
            json=weights_data
        )
        return response.status_code == 200
    except:
        print("Failed to send weights to server")
        return False

def get_global_weights():
    try:
        response = requests.get(f"{AGGREGATION_SERVER}/global_weights")
        if response.status_code == 200:
            global_weights = response.json()['weights']
            # Convert back to numpy arrays
            global_weights = [np.array(w) for w in global_weights]
            return global_weights
        return None
    except:
        print("Failed to get global weights from server")
        return None

def run_vehicle_simulation():
    print(f"Starting vehicle {VEHICLE_ID} simulation...")
    
    metrics = data_processor.get_network_metrics()
    print(f"Initial metrics: PDR={metrics['avg_pdr']:.3f}, "
          f"Latency={metrics['avg_latency']:.3f}ms, "
          f"Throughput={metrics['avg_throughput']:.3f}KBps")
    
    for episode in range(EPISODES):
        total_reward = train_local_model()
        
        print(f"Vehicle {VEHICLE_ID}, Episode {episode}: total_reward={total_reward}")
        
        if episode % FED_UPDATE_FREQUENCY == 0:
            send_weights_to_server()
            
            global_weights = get_global_weights()
            if global_weights:
                agent.set_weights(global_weights)
                print(f"Vehicle {VEHICLE_ID}: Updated with global weights")
    
    agent.save(f"vehicle_{VEHICLE_ID}_model.h5")
    
    metrics = data_processor.get_network_metrics()
    print(f"Final metrics: PDR={metrics['avg_pdr']:.3f}, "
          f"Latency={metrics['avg_latency']:.3f}ms, "
          f"Throughput={metrics['avg_throughput']:.3f}KBps")

if __name__ == "__main__":
    run_vehicle_simulation()