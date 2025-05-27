# sensor_graph_trajectory_definition.py  
import json
import time
import sys
import os
import cv2
# import subprocess
import numpy as np
import paho.mqtt.client as mqtt
from PyQt5.QtWidgets import QApplication, QDialog # QDialog for potentially non-blocking
from PyQt5.QtCore import QTimer # For non-blocking UI updates if needed
 
from graph_definition.graph_editor_ui import GraphEditor
from map_coverage_calculator.build_vision_cone import CameraCone
from graph_definition.sensor_graph_coverage_optimizer import CameraCoverageOptimizer
from graph_definition.sensor_graph_optimization import load_graph_and_roi, interpolate_graph_nodes, generate_candidate_configs, run_greedy_set_cover_with_visualization, save_optimized_cameras_to_json
 
STAGE = "sensor_graph_trajectory_definition"
 
# Ensure directories exist
os.makedirs('./graph_definition/graph_output/base_graph_from_human_input', exist_ok=True)
os.makedirs('./graph_definition/graph_output/coverage_subsets', exist_ok=True)
os.makedirs('./camera_poses_OPT_output', exist_ok=True)
 
# Create a dummy graph_and_roi.json if it doesn't exist, for the UI to load something
# In a real scenario, this would be output by a previous step or the UI itself would create it.
dummy_json_graph_path = './graph_definition/graph_output/base_graph_from_human_input/graph_and_roi.json'
if not os.path.exists(dummy_json_graph_path):
    print(f"[{STAGE.upper()}] Creating dummy '{dummy_json_graph_path}' as it's missing.")
    dummy_data = {
        "nodes": [], "edges": [], "roi": [], "reference_point": None, "map_path": None, "map_scale": 0.05
    }
    with open(dummy_json_graph_path, 'w') as f:
        json.dump(dummy_data, f)
 
 
def on_message(client, userdata, msg):
    payload = json.loads(msg.payload.decode())
    job_id = payload["job_id"]
    print(f"[{STAGE.upper()}] Starting job {job_id}")
 
    reference_point = None  # TODO: retrieval of reference point
 
    # I. Graph Editor UI
    # QApplication.instance() returns the current instance, or None if no instance exists.
    # This is important if on_message could be called multiple times.
    app = QApplication.instance()
    if app is None: # Create QApplication if it doesn't exist
        app = QApplication(sys.argv)
    
    # Check if map image exists for GraphEditor
    # The GraphEditor expects the map to be defined in the graph_and_roi.json or passed.
    # For now, let's assume the default behavior of GraphEditor handles map loading.
    # It might load from a fixed path or from what's in graph_and_roi.json
    
    # A default map path for the editor if not specified in graph_and_roi.json
    # This path should match what GraphEditor expects or be configurable.
    default_map_for_editor = './map_coverage_calculator/occumap_warehouse_5cm.png'
    if not os.path.exists(default_map_for_editor):
        print(f"[{STAGE.upper()}] WARNING: Default map for editor '{default_map_for_editor}' not found. UI might not show map.")
        # Create a dummy map if it's critical for UI to start
        # dummy_map = np.zeros((100,100,3), dtype=np.uint8)
        # cv2.imwrite(default_map_for_editor, dummy_map)
 
    ex = GraphEditor(map_scale=0.05) # Pass the path
    ex.show()
    print(f"[{STAGE.upper()}] Displaying GraphEditor UI. Close UI to continue...")
    app.exec_() # This is BLOCKING. MQTT connection might time out if UI is open too long.
    print(f"[{STAGE.upper()}] GraphEditor UI closed.")
 
 
    # II. Sensor Graph Trajectory Optimization
    json_graph_path = './graph_definition/graph_output/base_graph_from_human_input/graph_and_roi.json'
    map_image_path = './map_coverage_calculator/occumap_warehouse_5cm.png'
    image_output_path = './graph_definition/graph_output/coverage_subsets'
 
    if not os.path.exists(json_graph_path):
        print(f"[{STAGE.upper()}] ERROR: Graph file '{json_graph_path}' not found after UI. Cannot proceed with optimization.")
        # Publish error or handle
        response = {"status": "error", "timestamp": time.time(), "details": f"Graph file not found: {json_graph_path}"}
    elif not os.path.exists(map_image_path):
        print(f"[{STAGE.upper()}] ERROR: Map image '{map_image_path}' not found. Cannot proceed with optimization.")
        response = {"status": "error", "timestamp": time.time(), "details": f"Map image not found: {map_image_path}"}
    else:
        print(f"[{STAGE.upper()}] Starting optimization...")
        map_image = cv2.imread(map_image_path)
        if map_image is None:
            print(f"[{STAGE.upper()}] ERROR: Could not read map image '{map_image_path}'.")
            response = {"status": "error", "timestamp": time.time(), "details": f"Could not read map image: {map_image_path}"}
        else:
            image_height, image_width = map_image.shape[:2]
            nodes, region_info, reference_point_loaded = load_graph_and_roi(json_graph_path)
            
            # Use reference_point_loaded from the JSON if available, otherwise the initial one
            if reference_point_loaded:
                reference_point = reference_point_loaded
 
            resolution = 1 / 0.05 # pixels per meter
 
            focal_length = 18.5
            horizontal_aperture = 36.0
            vertical_aperture = 10.42
            height = 1.45
            horizontal_fov = 2 * np.arctan(horizontal_aperture / (2 * focal_length))
            vertical_fov = 2 * np.arctan(vertical_aperture / (2 * focal_length))
            z_min = height * np.tan(np.abs(vertical_fov / 2)) # Ensure positive z_min
            z_max = 16
 
            sensor = CameraCone(horizontal_fov, z_min, z_max, resolution)
            _, visible_points = sensor.find_visible_points()
 
            nodes_interpolated = interpolate_graph_nodes(
                nodes, map_scale=0.05, max_distance_m=2.0,
                min_distance_m=0.5, max_interpolations_per_edge=5
            )
            
            candidate_configs = generate_candidate_configs(
                nodes_interpolated, map_scale=0.05, image_height=image_height,
                image_width=image_width, reference_point=reference_point
            )
 
            optimizer = CameraCoverageOptimizer(
                map_image=map_image, region_info=region_info,
                resolution=resolution, visible_points=visible_points
            )
 
            optimized_camera_list = run_greedy_set_cover_with_visualization(
                optimizer=optimizer, candidate_configs=candidate_configs,
                region_info=region_info, visible_points=visible_points,
                resolution=resolution, image_output_path=image_output_path,
                coverage_threshold=0.95, verbose=False
            )
 
            output_path = './camera_poses_OPT_output'
            output_json_path = os.path.join(output_path, "optimized_cameras.json")
            save_optimized_cameras_to_json(optimized_camera_list, output_json_path, resolution, reference_point)
            print(f"[{STAGE.upper()}] Optimization complete. Results saved to {output_json_path}")
            response = {
                "status": "done",
                "timestamp": time.time(),
                "details": f"{STAGE} graph creation and optimization completed",
                "optimized_cameras_path": output_json_path
            }
 
    result_topic = f"pipeline/{STAGE}/status/{job_id}"
    
    # REMOVED: client.loop(timeout=1.0) - Not needed with loop_forever() and harmful here.
    
    print(f"[{STAGE.upper()}] Publishing result to {result_topic}: {json.dumps(response)}")
    info = client.publish(result_topic, json.dumps(response), qos=1) # Use QoS 1 for more reliability
 
    # Correct way to check if published for QoS 1 with loop_forever()
    # is_published() will be updated by the network loop.
    # We can wait for the publish callback or check rc.
    # For QoS 1 and 2, publish() will block until the message is Pipelined through the MQTT client’s network buffer.
    # It does not block until the PUBACK is received. The PUBACK is handled in the network loop.
    # info.wait_for_publish(timeout=5) # This is a blocking call, useful if not in callback.
 
    # Check info.rc immediately. For QoS 1, MQTT_ERR_SUCCESS means it's queued.
    if info.rc == mqtt.MQTT_ERR_SUCCESS:
        print(f"[{STAGE.upper()}] Publish initiated successfully (rc=0). Message ID: {info.mid}")
        # You can optionally wait for the publish to complete using a timeout,
        # but be careful not to block the loop for too long.
        # A robust way is to use `on_publish` callback.
        # For simplicity here, we'll assume loop_forever() handles it.
        # The `is_published()` check with `time.sleep()` was problematic.
        # If you need to confirm, use on_publish callback or info.wait_for_publish()
        # but wait_for_publish() will block.
 
        # A simple non-blocking check (might not be fully confirmed but better than sleep):
        # start_time = time.time()
        # while not info.is_published():
        #     if time.time() - start_time > 2.0: # 2 second timeout
        #         print(f"[{STAGE.upper()}] Publish confirmation timed out (rc={info.rc}, mid={info.mid}).")
        #         break
        #     # app.processEvents() # If you were in a tight loop *without* time.sleep, to keep UI responsive
        #     # No client.loop() here because loop_forever is running
        # if info.is_published():
        #    print(f"[{STAGE.upper()}] Publish confirmed (rc={info.rc}, mid={info.mid}).")
 
    else:
        print(f"[{STAGE.upper()}] Failed to initiate publish (rc={info.rc}). Message not sent.")
 
    print(f"[{STAGE.upper()}] Job {job_id} processing finished in on_message.")
 
# --- Global MQTT Client Setup ---
client = mqtt.Client(client_id=f"client-{STAGE}-{os.getpid()}") # Unique client ID
client.on_message = on_message
 
# Optional: Add on_publish callback for better publish confirmation
def on_publish(client, userdata, mid):
    print(f"[{STAGE.upper()}] Message published with MID {mid}")
client.on_publish = on_publish
 
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"[{STAGE.upper()}] Connected to MQTT Broker!")
        client.subscribe(f"pipeline/{STAGE}/start/+")
        print(f"[{STAGE.upper()}] Subscribed to topic: pipeline/{STAGE}/start/+")
    else:
        print(f"[{STAGE.upper()}] Failed to connect, return code {rc}\n")
client.on_connect = on_connect
 
def on_disconnect(client, userdata, rc):
    print(f"[{STAGE.upper()}] Disconnected from MQTT Broker with result code {rc}.")
    if rc != 0:
        print(f"[{STAGE.upper()}] Unexpected disconnection.")
        # Implement reconnection logic if desired, though loop_forever handles some of this.
 
client.on_disconnect = on_disconnect
 
 
try:
    print(f"[{STAGE.upper()}] Connecting to MQTT broker...")
    client.connect("localhost", 1883, keepalive=60) # keepalive is in seconds
except ConnectionRefusedError:
    print(f"[{STAGE.upper()}] MQTT connection refused. Is the broker running at localhost:1883?")
    sys.exit(1)
except Exception as e:
    print(f"[{STAGE.upper()}] MQTT connection error: {e}")
    sys.exit(1)
 
 
print(f"[{STAGE.upper()}] Starting MQTT client loop_forever()...")
# loop_forever() is blocking. It will run until client.disconnect() is called or an error.
client.loop_forever()