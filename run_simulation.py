# run_simulation.py
import pandas as pd
import numpy as np
import os
import subprocess
from vm_deployment import VehicleVM

def prepare_data_files():
    vehicle_data = pd.read_csv("v2v_vehicles.csv")
    communication_data = pd.read_csv("v2v_network.csv")

def start_aggregation_server():
    """Start the federated learning aggregation server"""
    subprocess.Popen(["python", "aggregation_server.py"])

def deploy_vehicle_vms(num_vehicles, num_rsus):
    """Deploy virtual machines for each vehicle"""
    for vehicle_id in range(num_vehicles):
        vm = VehicleVM(vehicle_id, num_vehicles, num_rsus)
        vm.create_dockerfile()
        vm.build_and_run()

def main():
    NUM_VEHICLES = 5  
    NUM_RSUS = 9
    
    prepare_data_files()
    
    start_aggregation_server()
    
    import time
    time.sleep(2)
    
    deploy_vehicle_vms(NUM_VEHICLES, NUM_RSUS)
    
    print(f"Simulation started with {NUM_VEHICLES} vehicles and {NUM_RSUS} RSUs")
    print("Check Docker containers for vehicle simulations")
    print("Use 'docker logs vehicle_X' to view progress")

if __name__ == "__main__":
    main()