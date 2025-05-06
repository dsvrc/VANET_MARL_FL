# run_simulation.sh
#!/bin/bash

# Number of vehicles to simulate
NUM_VEHICLES=5

# Start the server in the background
python run_server.py &
SERVER_PID=$!

# Wait for server to initialize
echo "Waiting for server to initialize..."
sleep 3

# Launch vehicles
python launch_vehicles.py $NUM_VEHICLES

# Kill the server when done
kill $SERVER_PID

echo "Simulation complete"