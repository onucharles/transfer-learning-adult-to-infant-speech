"""
Utility functions for I/O operations.
"""

import os
import json
import time


def create_folder(newpath):
    if not os.path.exists(newpath):
        os.makedirs(newpath)
        print("created directory: " + str(newpath))

def save_json(data, file_path):
    with open(file_path, 'w') as fp:
        json.dump(data, fp, sort_keys=True, indent=4)

def load_json(file_path):
    with open(file_path, 'r') as fp:
        data = json.load(fp)

    return data

def current_datetime():
    timestr = time.strftime("%Y%m%d-%H%M%S")
    return timestr

