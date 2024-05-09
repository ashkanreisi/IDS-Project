from ultralytics import YOLO
import sys, yaml
import os

os.environ['WANDB_MODE'] = 'online'

# Initialize an empty dictionary to store argument keys and values
args_dict = {}
# Define hyperparameter clamping ranges
clamping_ranges = {
        'lrf': (0.01, 1.0),
        'weight_decay': (0.0, 0.001),
        'warmup_epochs': (0.0, 5.0),
        'warmup_momentum': (0.0, 0.95),
        'box': (0.02, 0.2),
        'cls': (0.2, 4.0),
        'degrees': (0.0, 45.0),
        'translate': (0.0, 0.9),
        'scale': (0.0, 0.9),
        'shear': (0.0, 10.0),
        'perspective': (0.0, 0.001),
    }
# Iterate over each argument in sys.argv (skipping the script name)
for arg in sys.argv[1:]:
    # Split the argument on the first equals sign
    key, value = arg.split('=', 1)
    # Remove the leading '--' from the key
    key = key.lstrip('--')
    # Add the key-value pair to the dictionary
    if key in clamping_ranges:
        args_dict[key] = value

# Clamping hyperparameter values inside their accepted ranges
def clamp_value(value, min_value, max_value):
    return max(min_value, min(value, max_value))

# Convert hyperparameters to floats and clamp values as needed, skipping the 'data' key
args_dict = {
    key: clamp_value(float(value), *clamping_ranges[key]) if key in clamping_ranges else float(value)
    for key, value in args_dict.items()
}
print("ARGS: ")
print(args_dict)

# Start tuning
model = YOLO('yolov9e.pt')
model.train(**args_dict,
            data='data.yaml',
            epochs=100,
            batch=16,
            patience=10,
            cache=False,
            plots=True,
            save=True,
            val=True,
            device=0)
