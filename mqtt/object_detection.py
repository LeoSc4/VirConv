# object_detection.py 
import os
import json
import subprocess
from pathlib import Path
import time
import datetime
import threading
import sys

import paho.mqtt.client as mqtt
from tools.pipeline_inference_object_detection import main

STAGE = "object_detection"

# GLOBAL MQTT client (für publish im Thread)
client = mqtt.Client()

def handle_object_detection(payload):
    job_id = payload["job_id"]
    print(f"[{STAGE.upper()}] Starting job {job_id}")

    try:
        # Step 0: Install package in development mode
        print("[INFO] Step 0: Install package in development mode")
        subprocess.run(["python3", "setup.py", "develop"], check=True)

        # Step 1: Run PENet for depth completion
        print("[INFO] Step 1: Run PENet for depth completion")
        penet_dir = "tools/PENet"
        print(f"[INFO] Current working directory: {os.getcwd()}")
        subprocess.run(["python3", "main.py", "--detpath", "../../data/kitti/training"], cwd=penet_dir, check=True)
        subprocess.run(["python3", "main.py", "--detpath", "../../data/kitti/testing"], cwd=penet_dir, check=True)

        # Step 2: Create dataset infos
        print("[INFO] Step 2: Creating dataset infos")
        subprocess.run([
            "python3", "-m", "pcdet.datasets.kitti.kitti_dataset_mm",
            "create_adtc_infos", "tools/cfgs/dataset_configs/IW-dataset-9.yaml"
        ], check=True)

        # Step 3: Run inference with VirConv model
        print("[INFO] Step 3: Run inference with VirConv model")
        log_dir = 'inference_logs/iw_data9'
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        log_file = Path(log_dir) / ('%s_log_inference.txt' % datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

        model_ckpt = 'output/models/kitti/VirConv-T-IW-DS-9/IW_DS9_60EP_AUG_SWP_DS9_AUG_krx6i4fc/ckpt/checkpoint_epoch_51.pth'
        point_cloud_range = None
        bbox_analysis_path = None

        sys.argv = [
            'tools/iw_inference_and_vis.py',
            '--cfg_file', 'tools/cfgs/models/kitti/VirConv-T-IW-DS-9.yaml',
            '--batch_size', '1',
            '--workers', '0'
        ]

        main(log_file, model_ckpt, point_cloud_range, bbox_analysis_path)

        # Final MQTT publish
        print(f"[{STAGE.upper()}] MQTT client connected: {client.is_connected()}")

        result_topic = f"pipeline/{STAGE}/status/{job_id}"
        response = {
            "status": "done",
            "timestamp": time.time(),
            "details": f"{STAGE} object detection completed",
        }

        print(f"[{STAGE.upper()}] Publishing result to {result_topic}")
        info = client.publish(result_topic, json.dumps(response), qos=0)
        info.wait_for_publish()

        print(f"[{STAGE.upper()}] Publish success: {info.is_published()}, rc = {info.rc}")
        print("-------------- DONE -------------")
    except Exception as e:
        print(f"[{STAGE.upper()}] ERROR in handle_object_detection: {e}")
        import traceback
        traceback.print_exc()

def on_message(mqtt_client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        # Starte Thread mit der Jobverarbeitung
        thread = threading.Thread(target=handle_object_detection, args=(payload,))
        thread.start()
    except Exception as e:
        print(f"[{STAGE.upper()}] ERROR in on_message: {e}")

# MQTT setup
client.on_message = on_message
client.connect("127.0.0.1", 1883)  # Docker → Host MQTT Broker
client.subscribe(f"pipeline/{STAGE}/start/+")
client.loop_start()  # Netzwerkloop im Hintergrund

print(f"[{STAGE.upper()}] Listening to topic: pipeline/{STAGE}/start/+")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Stopping...")
    client.loop_stop()
    client.disconnect()
