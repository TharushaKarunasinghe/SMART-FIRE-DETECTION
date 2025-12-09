"""
Configuration settings for Smart Fire Detection System
"""

# File paths
RAW_DATA_DIR = "data/raw/"
PROCESSED_DATA_DIR = "data/processed/"
MODEL_DIR = "data/models/"

# Dataset file names (update after downloading)
SMOKE_DATASET = "smoke_detection_iot.csv"
ENV_DATASET = "environmental_sensor.csv"
SENSOR_FUSION_DATASET = "sensor_fusion_smoke.csv"

# Feature columns
FEATURE_COLUMNS = ['temperature', 'humidity', 'smoke', 'gas']

# Warning level thresholds
WARNING_LEVELS = {
    0: 'All Clear',
    1: 'Watch',
    2: 'Caution',
    3: 'Warning',
    4: 'Emergency'
}

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_DEPTH = 8  # For Decision Tree (NodeMCU memory constraint)
MIN_SAMPLES_SPLIT = 100
MIN_SAMPLES_LEAF = 50

# Target data points
TARGET_SAMPLES = 50000
