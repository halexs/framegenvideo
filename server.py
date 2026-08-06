import subprocess
import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

INPUT = "input.mp4"
HLS_DIR = "hls"

app = FastAPI()


os.makedirs(HLS_DIR, exist_ok=True)


@app.on_event("startup")
def startup():

    # subprocess.Popen([
    #     "ffmpeg",
    #     "-y",
    #     # "-hwaccel", "cuda",
    #     "-re",
    #     "-i", INPUT,

    #     "-c:v", "libx264",
    #     "-preset", "veryfast",

    #     "-f", "hls",

    #     "-hls_time", "2",
    #     "-hls_list_size", "5",
    #     "-hls_flags", "delete_segments",

    #     f"{HLS_DIR}/stream.m3u8"
    # ])

    subprocess.Popen([
        "ffmpeg",
        "-y",
        "-hwaccel", "cuda",
        # "-re",
        "-i", INPUT,

        "-c:v", "libx264",
        "-preset", "veryfast",

        "-f", "hls",

        "-hls_time", "4",
        "-hls_list_size", "0",

        f"{HLS_DIR}/stream.m3u8"
    ])


app.mount(
    "/hls",
    StaticFiles(directory=HLS_DIR),
    name="hls"
)


@app.get("/")
def home():

    return HTMLResponse("""
<!DOCTYPE html>
<html>
<head>
    <title>StreamerFrames</title>
</head>

<body>

<h1>StreamerFrames</h1>

<video
    id="video"
    controls
    autoplay
    width="800">
</video>


<script src="https://cdn.jsdelivr.net/npm/hls.js@latest"></script>

<script>

let video = document.getElementById("video");

let url = "/hls/stream.m3u8";

if (Hls.isSupported()) {

    let hls = new Hls();

    hls.loadSource(url);
    hls.attachMedia(video);

}
else if (video.canPlayType("application/vnd.apple.mpegurl")) {

    video.src = url;

}

</script>


</body>
</html>
""")