import subprocess, os

def run_cmd(cmd):
    print('RUN:', cmd)
    proc = subprocess.run(cmd, shell=True)
    return proc.returncode

def extract_frames(video_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    # 1-based frame numbers for ffmpeg naming
    cmd = f'ffmpeg -y -i "{video_path}" -qscale:v 2 "{out_dir}/frame_%06d.png"'
    return run_cmd(cmd)

def encode_side_by_side(left_pattern, right_pattern, out_path, fps=24):
    # left_pattern, right_pattern must match same frames
    # Use hstack to create side-by-side video
    cmd = f'ffmpeg -y -r {fps} -i "{left_pattern}" -r {fps} -i "{right_pattern}" -filter_complex "[0][1]hstack=inputs=2" -c:v libx264 -crf 18 "{out_path}"'
    return run_cmd(cmd)
