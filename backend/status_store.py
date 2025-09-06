import json, os

def write_status(job_dir, data):
    os.makedirs(job_dir, exist_ok=True)
    path = os.path.join(job_dir, 'status.json')
    with open(path, 'w') as f:
        json.dump(data, f)

def read_status(job_dir):
    path = os.path.join(job_dir, 'status.json')
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        return json.load(f)
