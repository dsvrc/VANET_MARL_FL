# data_processor.py
import pandas as pd
import numpy as np

class VANETDataProcessor:
    def __init__(self):
        self.vehicle_data = None
        self.communication_data = None
        
    def load_data(self, vehicle_file, communication_file):
        self.vehicle_data = pd.read_csv(vehicle_file)
        self.communication_data = pd.read_csv(communication_file)
        
    def get_vehicle_states(self, timestamp=0.0):
        """Return vehicle states at a specific timestamp"""
        return self.vehicle_data[self.vehicle_data['timestamp'] == timestamp]
    
    def get_communication_links(self, timestamp=0.0):
        """Return communication links at a specific timestamp"""
        return self.communication_data[self.communication_data['timestamp'] == timestamp]
    
    def get_vehicle_neighbors(self, vehicle_id, timestamp=0.0):
        """Get neighbors for a specific vehicle"""
        links = self.get_communication_links(timestamp)
        neighbors = links[links['source_id'] == vehicle_id]['target_id'].tolist()
        return neighbors
    
    def get_network_metrics(self, timestamp=0.0):
        """Calculate network metrics at a timestamp"""
        links = self.get_communication_links(timestamp)
        avg_pdr = links['pdr'].mean()
        avg_latency = links['latency_ms'].mean()
        packet_size_kb = 1.0
        avg_throughput = avg_pdr * packet_size_kb / (avg_latency / 1000) # in KBps
        
        return {
            'avg_pdr': avg_pdr,
            'avg_latency': avg_latency,
            'avg_throughput': avg_throughput
        }