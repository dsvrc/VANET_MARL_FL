# launch_vehicles.py
import subprocess
import time
import sys

def launch_vehicles(num_vehicles):
    processes = []
    
    print(f"Launching {num_vehicles} vehicle instances...")
    
    for vehicle_id in range(num_vehicles):
        cmd = [sys.executable, "vehicle.py", str(vehicle_id)]
        proc = subprocess.Popen(cmd)
        processes.append(proc)
        print(f"Started vehicle {vehicle_id}")
        time.sleep(1)  # Stagger vehicle startup
    
    print(f"All {num_vehicles} vehicles launched")
    
    try:
        for proc in processes:
            proc.wait()
    except KeyboardInterrupt:
        print("Stopping all vehicles...")
        for proc in processes:
            proc.terminate()

if __name__ == "__main__":
    num_vehicles = 5  # Default number of vehicles
    
    if len(sys.argv) > 1:
        num_vehicles = int(sys.argv[1])
    
    launch_vehicles(num_vehicles)