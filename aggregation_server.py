# aggregation_server.py
from flask import Flask, request, jsonify
import numpy as np
from federated_learning import FederatedLearning

app = Flask(__name__)

NUM_VEHICLES = 5 
fed_learning = FederatedLearning(NUM_VEHICLES)

@app.route('/update_weights', methods=['POST'])
def update_weights():
    """Receive updated weights from vehicles"""
    data = request.json
    vehicle_id = data['vehicle_id']
    weights = data['weights']
    
    weights = [np.array(w) for w in weights]
    
    fed_learning.update_vehicle_model(vehicle_id, weights)
    
    if len(fed_learning.vehicle_weights) == NUM_VEHICLES:
        fed_learning.aggregate_models()
        
    return jsonify({'status': 'success'})

@app.route('/global_weights', methods=['GET'])
def get_global_weights():
    """Return global model weights"""
    if fed_learning.global_weights is None:
        return jsonify({'error': 'Global model not initialized'}), 404
        
    weights = [w.tolist() for w in fed_learning.global_weights]
    
    return jsonify({'weights': weights})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)