#!/bin/bash

# This script should first run the graph_editor_ui to start the UI and create new graph.json 

python3 graph_editor_ui.py

# Run the sensor_graph_optimization.py script with the generated graph.json

python3 sensor_graph_optimization.py 