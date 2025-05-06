# run_server.py
from server import Server

if __name__ == "__main__":
    num_vehicles = 5  # Adjust as needed
    action_size = 7   # IMPORTANT: Must match the vehicles' environment action space
    
    print(f"Starting server for {num_vehicles} vehicles with action_size={action_size}")
    server = Server(num_vehicles=num_vehicles, action_size=action_size)
    server.start()