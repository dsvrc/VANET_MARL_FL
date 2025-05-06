# vm_deployment.py
import os
import subprocess

class VehicleVM:
    def __init__(self, vehicle_id, num_vehicles, num_rsus):
        self.vehicle_id = vehicle_id
        self.num_vehicles = num_vehicles
        self.num_rsus = num_rsus
        
    def create_dockerfile(self):
        dockerfile = f"""
FROM python:3.8-slim

RUN pip install numpy pandas tensorflow

WORKDIR /app

COPY data_processor.py /app/
COPY vanet_env.py /app/
COPY rl_agent.py /app/
COPY federated_learning.py /app/
COPY vehicle_simulation.py /app/
COPY vehicle_data.csv /app/
COPY communication_data.csv /app/

ENV VEHICLE_ID={self.vehicle_id}
ENV NUM_VEHICLES={self.num_vehicles}
ENV NUM_RSUS={self.num_rsus}

CMD ["python", "vehicle_simulation.py"]
"""
        with open(f"Dockerfile_{self.vehicle_id}", "w") as f:
            f.write(dockerfile)
            
    def build_and_run(self):
        
        subprocess.run(["docker", "build", "-t", f"vehicle_{self.vehicle_id}", 
                        "-f", f"Dockerfile_{self.vehicle_id}", "."])
        
        # Run Docker container
        subprocess.run(["docker", "run", "-d", "--name", f"vehicle_{self.vehicle_id}", 
                        f"vehicle_{self.vehicle_id}"])