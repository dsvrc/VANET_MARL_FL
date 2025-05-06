# server.py
import socket
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import threading
import pickle
import time

class Server:
    def __init__(self, host='0.0.0.0', base_port=5000, num_vehicles=5, state_size=20, action_size=7):
        self.host = host
        self.base_port = base_port
        self.num_vehicles = num_vehicles
        self.state_size = state_size
        self.action_size = action_size  # IMPORTANT: Match with vehicle environment
        self.vehicle_connections = {}
        self.vehicle_weights = {}
        self.global_model = self._build_model()
        self.stop_flag = False
        
    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))  # Match with vehicle action_size
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
        return model
        
    def start(self):
        print(f"Server starting... Listening on ports {self.base_port} to {self.base_port + self.num_vehicles - 1}")
        print(f"Server model action size: {self.action_size}")
        
        for vehicle_id in range(self.num_vehicles):
            port = self.base_port + vehicle_id
            thread = threading.Thread(target=self._handle_vehicle, args=(vehicle_id, port))
            thread.daemon = True
            thread.start()
            
        # Start the aggregation thread
        agg_thread = threading.Thread(target=self._periodic_aggregation)
        agg_thread.daemon = True
        agg_thread.start()
        
        try:
            while not self.stop_flag:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Server shutting down...")
            self.stop_flag = True
    
    def _handle_vehicle(self, vehicle_id, port):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((self.host, port))
        server_socket.listen(1)
        
        print(f"Listening for vehicle {vehicle_id} on port {port}")
        
        conn, addr = server_socket.accept()
        print(f"Vehicle {vehicle_id} connected from {addr}")
        self.vehicle_connections[vehicle_id] = conn
        
        try:
            while not self.stop_flag:
                # Receive data size first
                data_size_bytes = conn.recv(4)
                if not data_size_bytes:
                    break
                    
                data_size = int.from_bytes(data_size_bytes, byteorder='big')
                
                # Receive action size first to ensure model compatibility
                action_size_bytes = conn.recv(4)
                if not action_size_bytes:
                    break
                
                vehicle_action_size = int.from_bytes(action_size_bytes, byteorder='big')
                
                if vehicle_action_size != self.action_size:
                    print(f"Warning: Vehicle {vehicle_id} has action_size {vehicle_action_size}, server has {self.action_size}")
                    # Option: Rebuild server model if needed
                    # self.action_size = vehicle_action_size
                    # self.global_model = self._build_model()
                
                # Receive actual weights data
                data = b''
                remaining = data_size
                while remaining > 0:
                    chunk = conn.recv(min(4096, remaining))
                    if not chunk:
                        break
                    data += chunk
                    remaining -= len(chunk)
                
                if not data:
                    break
                
                try:
                    weights = pickle.loads(data)
                    self.vehicle_weights[vehicle_id] = weights
                    print(f"Received weights from vehicle {vehicle_id}")
                    
                    # Send global model back immediately
                    global_weights = self.global_model.get_weights()
                    serialized_weights = pickle.dumps(global_weights)
                    size_bytes = len(serialized_weights).to_bytes(4, byteorder='big')
                    conn.sendall(size_bytes + serialized_weights)
                    
                except Exception as e:
                    print(f"Error processing data from vehicle {vehicle_id}: {e}")
                    
        except Exception as e:
            print(f"Connection error with vehicle {vehicle_id}: {e}")
        finally:
            conn.close()
            if vehicle_id in self.vehicle_connections:
                del self.vehicle_connections[vehicle_id]
            print(f"Vehicle {vehicle_id} disconnected")
    
    def _periodic_aggregation(self):
        while not self.stop_flag:
            if len(self.vehicle_weights) >= max(1, self.num_vehicles // 2):
                print(f"Aggregating weights from {len(self.vehicle_weights)} vehicles")
                try:
                    self._aggregate_models()
                    print("Global model updated")
                except Exception as e:
                    print(f"Error during model aggregation: {e}")
                    # Clear problematic weights
                    self.vehicle_weights = {}
            
            time.sleep(5)  # Aggregate every 5 seconds if enough data
    
    def _aggregate_models(self):
        if not self.vehicle_weights:
            return
            
        # Verify all weight shapes match
        first_vehicle_id = list(self.vehicle_weights.keys())[0]
        first_weights = self.vehicle_weights[first_vehicle_id]
        
        for vehicle_id, weights in list(self.vehicle_weights.items()):
            if len(weights) != len(first_weights):
                print(f"Vehicle {vehicle_id} has different model structure. Skipping.")
                del self.vehicle_weights[vehicle_id]
                continue
                
            for i, layer_weights in enumerate(weights):
                if layer_weights.shape != first_weights[i].shape:
                    print(f"Vehicle {vehicle_id} has incompatible layer shape at layer {i}. Skipping.")
                    print(f"Expected {first_weights[i].shape}, got {layer_weights.shape}")
                    del self.vehicle_weights[vehicle_id]
                    break
        
        if not self.vehicle_weights:
            print("No compatible weights to aggregate")
            return
        
        # Get shapes from first vehicle
        shapes = [w.shape for w in list(self.vehicle_weights.values())[0]]
        
        # Initialize aggregated weights
        aggregated_weights = [np.zeros(shape) for shape in shapes]
        
        # Sum all weights
        for vehicle_id, weights in self.vehicle_weights.items():
            for i, layer_weights in enumerate(weights):
                aggregated_weights[i] += layer_weights
        
        # Average the weights
        for i in range(len(aggregated_weights)):
            aggregated_weights[i] /= len(self.vehicle_weights)
        
        # Update global model
        self.global_model.set_weights(aggregated_weights)
        
        # Clear the weights buffer
        self.vehicle_weights = {}

if __name__ == "__main__":
    server = Server(num_vehicles=5, action_size=7)  # Make sure action_size matches vehicle environment
    server.start()