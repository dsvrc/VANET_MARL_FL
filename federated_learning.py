# federated_learning.py
import numpy as np

class FederatedLearning:
    def __init__(self, num_vehicles):
        self.num_vehicles = num_vehicles
        self.global_weights = None
        self.vehicle_weights = {}
        
    def initialize_global_model(self, model_weights):
        """Initialize global model with weights"""
        self.global_weights = model_weights
        
    def update_vehicle_model(self, vehicle_id, model_weights):
        """Update weights for a specific vehicle"""
        self.vehicle_weights[vehicle_id] = model_weights
        
    def aggregate_models(self):
        """Aggregate models using FedAvg algorithm"""
        if not self.vehicle_weights:
            return self.global_weights
            
        aggregated_weights = []
        
        first_vehicle = list(self.vehicle_weights.keys())[0]
        for i in range(len(self.vehicle_weights[first_vehicle])):
            layer_shape = self.vehicle_weights[first_vehicle][i].shape
            layer_weights = np.zeros(layer_shape)
            
            for vehicle_id in self.vehicle_weights:
                layer_weights += self.vehicle_weights[vehicle_id][i]
                
            layer_weights /= len(self.vehicle_weights)
            aggregated_weights.append(layer_weights)
            
        self.global_weights = aggregated_weights
        return self.global_weights
    
    def distribute_global_model(self):
        """Distribute global model to all vehicles"""
        return self.global_weights