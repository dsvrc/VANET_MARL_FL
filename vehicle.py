# vehicle.py
import socket
import numpy as np
import pickle
import time
import json
import sys
import tensorflow as tf
from data_processor import VANETDataProcessor
from vanet_env import VANETEnvironment
from rl_agent import DQNAgent

class Vehicle:
    def __init__(self, vehicle_id, server_host='localhost', server_base_port=5000, 
                 num_vehicles=5, num_rsus=2):
        self.vehicle_id = vehicle_id
        self.server_host = server_host
        self.server_port = server_base_port + vehicle_id
        self.num_vehicles = num_vehicles
        self.num_rsus = num_rsus
        self.socket = None
        self.connected = False
        
        # Initialize environment and agent
        self.data_processor = VANETDataProcessor()
        self.data_processor.load_data('v2v_vehicles.csv', 'v2v_network.csv')
        
        self.env = VANETEnvironment(self.data_processor, num_vehicles, num_rsus)
        state_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.n
        
        print(f"Vehicle {vehicle_id} initialized with action_size: {action_size}")
        
        self.agent = DQNAgent(state_size, action_size)
        
    def connect_to_server(self):
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.server_host, self.server_port))
            self.connected = True
            print(f"Vehicle {self.vehicle_id} connected to server on port {self.server_port}")
            return True
        except Exception as e:
            print(f"Failed to connect: {e}")
            return False
    
    def send_weights(self):
        if not self.connected:
            print("Not connected to server")
            return False
            
        try:
            weights = self.agent.get_weights()
            serialized_weights = pickle.dumps(weights)
            
            # Send size first
            size_bytes = len(serialized_weights).to_bytes(4, byteorder='big')
            self.socket.sendall(size_bytes)
            
            # Send action size to ensure compatibility
            action_size_bytes = self.agent.action_size.to_bytes(4, byteorder='big')
            self.socket.sendall(action_size_bytes)
            
            # Send actual weights
            self.socket.sendall(serialized_weights)
            
            print(f"Vehicle {self.vehicle_id} sent weights to server (action_size: {self.agent.action_size})")
            return True
        except Exception as e:
            print(f"Failed to send weights: {e}")
            self.connected = False
            return False
    
    def receive_global_weights(self):
        if not self.connected:
            print("Not connected to server")
            return None
            
        try:
            # Receive data size first
            data_size_bytes = self.socket.recv(4)
            if not data_size_bytes:
                print("Connection closed by server")
                self.connected = False
                return None
                
            data_size = int.from_bytes(data_size_bytes, byteorder='big')
            
            # Receive actual data
            data = b''
            remaining = data_size
            while remaining > 0:
                chunk = self.socket.recv(min(4096, remaining))
                if not chunk:
                    break
                data += chunk
                remaining -= len(chunk)
            
            if not data:
                print("No data received")
                return None
                
            global_weights = pickle.loads(data)
            
            # Verify compatibility
            if len(global_weights) != len(self.agent.get_weights()):
                print("Received incompatible model structure")
                return None
                
            for i, layer_weights in enumerate(global_weights):
                local_shape = self.agent.get_weights()[i].shape
                if layer_weights.shape != local_shape:
                    print(f"Layer {i} shape mismatch: local {local_shape} vs global {layer_weights.shape}")
                    return None
                    
            print(f"Vehicle {self.vehicle_id} received compatible global weights")
            return global_weights
        except Exception as e:
            print(f"Failed to receive weights: {e}")
            self.connected = False
            return None
    
    def train_local_model(self, episodes=5, batch_size=32):
        print(f"Vehicle {self.vehicle_id} starting local training for {episodes} episodes")
        
        for episode in range(episodes):
            state = self.env.reset()
            state = np.reshape(state, [1, self.agent.state_size])
            total_reward = 0
            
            for time_step in range(100):  # Max 100 steps per episode
                action = self.agent.act(state)
                
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.agent.state_size])
                
                self.agent.remember(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
                
                if done:
                    break
            
            # Train the neural network
            if len(self.agent.memory) > batch_size:
                self.agent.replay(batch_size)
            
            print(f"Vehicle {self.vehicle_id}, Episode {episode}: total_reward={total_reward:.2f}")
            
        print(f"Vehicle {self.vehicle_id} completed local training")
    
    def run(self, total_rounds=10, episodes_per_round=5):
        if not self.connect_to_server():
            print("Exiting due to connection failure")
            return
            
        try:
            for round_num in range(total_rounds):
                print(f"\nVehicle {self.vehicle_id} starting round {round_num+1}/{total_rounds}")
                
                # 1. Train local model
                self.train_local_model(episodes=episodes_per_round)
                
                # 2. Send weights to server
                if not self.send_weights():
                    print("Failed to send weights, reconnecting...")
                    if not self.connect_to_server():
                        print("Reconnection failed, exiting")
                        break
                    continue
                
                # 3. Receive global weights
                global_weights = self.receive_global_weights()
                if global_weights is not None:
                    try:
                        self.agent.set_weights(global_weights)
                        print(f"Vehicle {self.vehicle_id} updated with global weights")
                    except Exception as e:
                        print(f"Error updating with global weights: {e}")
                else:
                    print("Failed to receive global weights")
                
                # Calculate and print metrics
                metrics = self.data_processor.get_network_metrics()
                print(f"Network metrics: PDR={metrics['avg_pdr']:.3f}, "
                      f"Latency={metrics['avg_latency']:.3f}ms, "
                      f"Throughput={metrics['avg_throughput']:.3f}KBps")
                
                # Short delay between rounds
                time.sleep(2)
                
            # Save final model
            self.agent.save(f"vehicle_{self.vehicle_id}_model.h5")
            print(f"Vehicle {self.vehicle_id} saved final model")
            
        except KeyboardInterrupt:
            print(f"Vehicle {self.vehicle_id} stopping")
        finally:
            if self.socket:
                self.socket.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python vehicle.py <vehicle_id>")
        sys.exit(1)
        
    vehicle_id = int(sys.argv[1])
    vehicle = Vehicle(vehicle_id)
    vehicle.run()