from server import list_video_paths, run_framegen, get_hls_dir, sanitize_name, generation_jobs, get_movies_root
import time, os
videos = list_video_paths()
print('found', len(videos), 'videos')
video = videos[0]
print('starting generation for', video)
run_framegen(video)
key = sanitize_name(str(video.relative_to(get_movies_root())))
print('job key', key)
# Wait briefly to allow process to create hls dir and files
time.sleep(5)
hls_dir = get_hls_dir(video)
print('hls_dir exists?', hls_dir.exists())
if hls_dir.exists():
    print('files:', os.listdir(hls_dir))
else:
    print('hls dir not present yet')
job = generation_jobs.get(key)
print('job present', job is not None)
if job:
    proc = job['process']
    print('pid', proc.pid, 'running?', proc.poll() is None)
    # fetch output so far
    try:
        out = proc.stderr.read(1024) if proc.stderr else b''
        print('stderr snippet:', out[:200])
    except Exception as e:
        print('could not read stderr:', e)
print('done')
