# framegenvideo

Local RIFE-based frame interpolation (e.g. 23.976 -> 60 fps) with a small FastAPI web UI
(`server.py`) that browses a movie folder and streams HLS while frames are generated.
Built on [Practical-RIFE](https://github.com/hzwer/Practical-RIFE) (MIT). See `PLAN.md` for the
audit and the implementation roadmap; `legacy/` holds superseded experiments.

Quick start (Windows): create a venv at `..\.framegen`, `pip install -r requirements.txt`,
put the RIFE model files in `train_log/`, then run `start_server.bat` (serves http://127.0.0.1:8000).
Set `MOVIES_DIR` (default `E:/Movies`), `HLS_DIR`, and for `rife_server.py` `HOST`/`PORT` to override paths.
The new `streamerframes/` package (see `README-StreamerFrames.md` and `PLAN.md`) has a CLI: `python -m streamerframes probe <video>` and
`python -m streamerframes plan <video> --target-fps 60` (output fps, frame count, segment layout), and
`python -m streamerframes check` (GPU, torch arch, ffmpeg/NVENC, model files),
`python -m streamerframes render <video> [--profile quality|realtime] [--target-fps 60 | --multi 2]` (resumable;
Ctrl+C pauses, rerun to continue; exports MP4/MKV with the source audio and subtitles), and
`python -m streamerframes cache {ls,rm,gc}`.
Configuration: copy `streamerframes.example.toml` to `streamerframes.toml`.
Run tests with `pip install pytest fastapi httpx && pytest`.
The server binds to localhost by default; change `--host` in the start scripts only if you trust your network.

---

# Practical-RIFE (upstream README)

# Practical-RIFE 
**[V4.0 Anime Demo Video](https://www.bilibili.com/video/BV1J3411t7qT?p=1&share_medium=iphone&share_plat=ios&share_session_id=7AE3DA72-D05C-43A0-9838-E2A80885BD4E&share_source=QQ&share_tag=s_i&timestamp=1639643780&unique_k=rjqO0EK)** | **[迭代经验](https://zhuanlan.zhihu.com/p/721430631)** | **[迭代QA](https://github.com/hzwer/Practical-RIFE/issues/124)** | **[Colab](https://colab.research.google.com/drive/10_n-qGYw5glYQ6k9cejUKr2MKFd5hssC?usp=sharing)**

This project is based on [RIFE](https://github.com/hzwer/arXiv2020-RIFE) and [SAFA](https://github.com/megvii-research/WACV2024-SAFA). We aim to enhance their practicality for users by incorporating various features and designing new models. Since improving the PSNR index is not consistent with subjective perception. This project is intended for engineers and developers. For general users, we recommend the following software:

**[SVFI (中文)](https://github.com/YiWeiHuang-stack/Squirrel-Video-Frame-Interpolation) | [RIFE-App](https://grisk.itch.io/rife-app) | [FlowFrames](https://nmkd.itch.io/flowframes)**

Thanks to [SVFI team](https://github.com/Justin62628/Squirrel-RIFE) to support model testing on Animation. 

[VapourSynth-RIFE](https://github.com/HolyWu/vs-rife) | [RIFE-ncnn-vulkan](https://github.com/nihui/rife-ncnn-vulkan) | [VapourSynth-RIFE-ncnn-Vulkan](https://github.com/styler00dollar/VapourSynth-RIFE-ncnn-Vulkan) | [vs-mlrt](https://github.com/AmusementClub/vs-mlrt) | [Drop frame fixer and FPS converter](https://github.com/may-son/RIFE-FixDropFrames-and-ConvertFPS)

## Frame Interpolation
2024.08 - We find that 4.24+ is quite suitable for post-processing of [some diffusion model generated videos](https://drive.google.com/drive/folders/1hSzUn10Era3JCaVz0Z5Eg4wT9R6eJ9U9?usp=sharing).

### Trained Model
The content of these links is under the same MIT license as this project. **lite** means using similar training framework, but lower computational cost model.
Currently, it is recommended to choose 4.25 by default for most scenes.

4.26 - 2024.09.21 | [Google Drive](https://drive.google.com/file/d/1gViYvvQrtETBgU1w8axZSsr7YUuw31uy/view?usp=sharing) [百度网盘](https://pan.baidu.com/s/1EZsG3IFO8C1e2uRVb_Npgg?pwd=smw8) || [4.25.lite - 2024.10.20](https://drive.google.com/file/d/1zlKblGuKNatulJNFf5jdB-emp9AqGK05/view?usp=share_link) 

4.25 - 2024.09.19 | [Google Drive](https://drive.google.com/file/d/1ZKjcbmt1hypiFprJPIKW0Tt0lr_2i7bg/view?usp=sharing) [百度网盘](https://pan.baidu.com/s/1rpUX5uawusz2uwEdXtjRbw?pwd=mo6k) | I am trying using more flow blocks, so the scale_list will change accordingly. It seems that the anime scenes have been significantly improved.

4.22 - 2024.08.08 | [Google Drive](https://drive.google.com/file/d/1qh2DSA9a1eZUTtZG9U9RQKO7N7OaUJ0_/view?usp=share_link)  [百度网盘](https://pan.baidu.com/s/1EA5BIHqOu35Rj4meg00G4g?pwd=hwym) || [4.22.lite](https://drive.google.com/file/d/1Smy6gY7BkS_RzCjPCbMEy-TsX8Ma5B0R/view?usp=sharing) || 4.21 - 2024.08.04 | [Google Drive](https://drive.google.com/file/d/1l5u6G8vEkPAT7cYYWwzB6OG8vwBYrxiS/view?usp=sharing) [百度网盘](https://pan.baidu.com/s/1TMjRFOwdLgsShKdGbTKW_g?pwd=4q6d)

4.20 - 2024.07.24 | [Google Drive](https://drive.google.com/file/d/11n3YR7-qCRZm9RDdwtqOTsgCJUHPuexA/view?usp=sharing) [百度网盘](https://pan.baidu.com/s/1v0b7ZTSj_VvLOfW-hQ_NZQ?pwd=ykkv)
|| 4.18 - 2024.07.03 | [Google Drive](https://drive.google.com/file/d/1octn-UVuEjXa_HlsIUbNeLTTvYCKbC_s/view?usp=sharing) [百度网盘](https://pan.baidu.com/s/1fqtxJyXSgUx-gE3rieuKxg?pwd=udr1)

4.17 - 2024.05.24 | [Google Drive](https://drive.google.com/file/d/1962p_lEWo_kLTEynarNaRYRNVdaiQG2k/view?usp=share_link) [百度网盘](https://pan.baidu.com/s/1bMzTYoJKZXsoxuSBmzj6VQ?pwd=as37) : Add gram loss from [FILM](https://github.com/google-research/frame-interpolation/blob/69f8708f08e62c2edf46a27616a4bfcf083e2076/losses/vgg19_loss.py) || [4.17.lite](https://drive.google.com/file/d/1e9Qb4rm20UAsO7h9VILDwrpvTSHWWW8b/view?usp=share_link)

4.15 - 2024.03.11 | [Google Drive](https://drive.google.com/file/d/1xlem7cfKoMaiLzjoeum8KIQTYO-9iqG5/view?usp=sharing) [百度网盘](https://pan.baidu.com/s/1IGNIX7JXGUwI_tfoafYHqA?pwd=bg0b) || [4.15.lite](https://drive.google.com/file/d/1BoOF-qSEnTPDjpKG1sBTa6k7Sv5_-k7z/view?usp=sharing) || 4.14 - 2024.01.08 | [Google Drive](https://drive.google.com/file/d/1BjuEY7CHZv1wzmwXSQP9ZTj0mLWu_4xy/view?usp=share_link) [百度网盘](https://pan.baidu.com/s/1d-W64lRsJTqNsgWoXYiaWQ?pwd=xawa) || [4.14.lite](https://drive.google.com/file/d/1eULia_onOtRXHMAW9VeDL8N2_7z8J1ba/view?usp=share_link)

v4.9.2 - 2023.11.01 | [Google Drive](https://drive.google.com/file/d/1UssCvbL8N-ty0xIKM5G5ZTEgp9o4w3hp/view?usp=sharing) [百度网盘](https://pan.baidu.com/s/18cbx3EP4HWgSa1vkcXvvyw?pwd=swr9) || v4.3 - 2022.8.17 | [Google Drive](https://drive.google.com/file/d/1xrNofTGMHdt9sQv7-EOG0EChl8hZW_cU/view?usp=sharing) [百度网盘](https://pan.baidu.com/s/12AUAeZLZf5E1_Zx6WkS3xw?pwd=q83a) 

v3.8 - 2021.6.17 | [Google Drive](https://drive.google.com/file/d/1O5KfS3KzZCY3imeCr2LCsntLhutKuAqj/view?usp=sharing) [百度网盘](https://pan.baidu.com/s/1X-jpWBZWe-IQBoNAsxo2mA?pwd=kxr3) || v3.1 - 2021.5.17 | [Google Drive](https://drive.google.com/file/d/1xn4R3TQyFhtMXN2pa3lRB8cd4E1zckQe/view?usp=sharing) [百度网盘](https://pan.baidu.com/s/1W4p_Ni04HLI_jTy45sVodA?pwd=64bz) 

[More Older Version](https://github.com/megvii-research/ECCV2022-RIFE/issues/41)

### Installation
python <= 3.11
```
git clone git@github.com:hzwer/Practical-RIFE.git
cd Practical-RIFE
pip3 install -r requirements.txt
```
Download a model from the model list and put *.py and flownet.pkl on train_log/
### Run

You can use our [demo video](https://drive.google.com/file/d/1i3xlKb7ax7Y70khcTcuePi6E7crO_dFc/view?usp=sharing) or your video. 
```
python3 inference_video.py --multi=2 --video=video.mp4 
```
(generate video_2X_xxfps.mp4)
```
python3 inference_video.py --multi=4 --video=video.mp4
```
(for 4X interpolation)
```
python3 inference_video.py --multi=2 --video=video.mp4 --scale=0.5
```
(If your video has high resolution, such as 4K, we recommend set --scale=0.5 (default 1.0))
```
python3 inference_video.py --multi=4 --img=input/
```
(to read video from pngs, like input/0.png ... input/612.png, ensure that the png names are numbers)

Parameter descriptions:

--img / --video: The input file address

--output: Output video name 'xxx.mp4'

--model: Directory with trained model files

--UHD: It is equivalent to setting scale=0.5

--montage: Splice the generated video with the original video, like [this demo](https://www.youtube.com/watch?v=kUQ7KK6MhHw)

--fps: Set output FPS manually

--ext: Set output video format, default: mp4

--multi: Interpolation frame rate multiplier

--exp: Set --multi to 2^(--exp)

--skip: It's no longer useful refer to [issue 207](https://github.com/hzwer/ECCV2022-RIFE/issues/207)


### Model training
The whole repo can be downloaded from [v4.0](https://drive.google.com/file/d/1zoSz7b8c6kUsnd4gYZ_6TrKxa7ghHJWW/view?usp=sharing), [v4.12](https://drive.google.com/file/d/1IHB35zhO4rr-JSMnpRvHhU9U65Z4giWv/view?usp=sharing), [v4.15](https://drive.google.com/file/d/19sUMZ-6H7g_hYDjTcqxYu9kE7TqnfS3k/view?usp=sharing), [v4.18](https://drive.google.com/file/d/1g8D2foww7DhGLIxtaDLr9fU3y-ByOw4B/view?usp=share_link), [v4.25](https://drive.google.com/file/d/1_l4OgBp3GrrHOcQB87xXCI7OtTzyeXZL/view?usp=share_link). However, we currently do not have the time to organize them well, they are for reference only.

## Video Enhancement

<img width="710" alt="image" src="https://github.com/hzwer/Practical-RIFE/assets/10103856/5bae134c-0747-4084-bbab-37b1595352f1">

We are developing a practical model of [SAFA](https://github.com/megvii-research/WACV2024-SAFA). Welcome to check its [demo](https://www.youtube.com/watch?v=QII2KQSBBwk) ([BiliBili](https://www.bilibili.com/video/BV1Up4y1d7kF/)) and provide advice.

v0.5 - 2023.12.26 | [Google Drive](https://drive.google.com/file/d/1OLO9hLV97ZQ4uRV2-aQqgnwhbKMMt6TX/view?usp=sharing)

```
python3 inference_video_enhance.py --video=demo.mp4
```

## Citation

```
@inproceedings{huang2022rife,
  title={Real-Time Intermediate Flow Estimation for Video Frame Interpolation},
  author={Huang, Zhewei and Zhang, Tianyuan and Heng, Wen and Shi, Boxin and Zhou, Shuchang},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2022}
}
```
```
@inproceedings{huang2024safa,
  title={Scale-Adaptive Feature Aggregation for Efficient Space-Time Video Super-Resolution},
  author={Huang, Zhewei and Huang, Ailin and Hu, Xiaotao and Hu, Chen and Xu, Jun and Zhou, Shuchang},
  booktitle={Winter Conference on Applications of Computer Vision (WACV)},
  year={2024}
}
```

## Reference

Optical Flow:
[ARFlow](https://github.com/lliuz/ARFlow)  [pytorch-liteflownet](https://github.com/sniklaus/pytorch-liteflownet)  [RAFT](https://github.com/princeton-vl/RAFT)  [pytorch-PWCNet](https://github.com/sniklaus/pytorch-pwc)

Video Interpolation: 
[DVF](https://github.com/lxx1991/pytorch-voxel-flow)  [TOflow](https://github.com/Coldog2333/pytoflow)  [SepConv](https://github.com/sniklaus/sepconv-slomo)  [DAIN](https://github.com/baowenbo/DAIN)  [CAIN](https://github.com/myungsub/CAIN)  [MEMC-Net](https://github.com/baowenbo/MEMC-Net)   [SoftSplat](https://github.com/sniklaus/softmax-splatting)  [BMBC](https://github.com/JunHeum/BMBC)  [EDSC](https://github.com/Xianhang/EDSC-pytorch)  [EQVI](https://github.com/lyh-18/EQVI) [RIFE](https://github.com/hzwer/arXiv2020-RIFE)

## Running the StreamerFrames server (Windows)

This repository includes a small FastAPI-based web server (server.py) and a RIFE HLS generator (rife_server.py). The server lists movies in E:\\Movies (or resolves the "Movies - Shortcut.lnk" file in the project root) and can play the original file or stream a RIFE-interpolated HLS version.

Quick start (uses the provided virtualenv at `.framegen`):

1. Open Command Prompt or PowerShell in the project folder.

2. Start the server with the bundled batch script (uses `.framegen` Python):

   - Double-click start_server.bat or run:

     start_server.bat

   - Or launch the PowerShell helper which opens a server window and your browser:

     start_server.ps1

3. The UI will be available at http://localhost:8000. Choose a movie and click "Normal" to play the original or "Framegen" to prepare and play the interpolated HLS stream.

Manual start using the project virtualenv Python:

- Run uvicorn directly from the venv:

  .\\.framegen\\Scripts\\python.exe -m uvicorn server:app --host 0.0.0.0 --port 8000

Generate a framegen HLS stream manually:

- Generate-only mode for testing (creates HLS in an output folder):

  .\\.framegen\\Scripts\\python.exe rife_server.py --input "E:\\Movies\\MyMovie.mp4" --hls-dir "hls/MyMovie" --generate-only

- The web UI will start generation automatically when you use the "Framegen" link for a movie.

Auto-start on logon (optional):

- Run create_schtask.bat as Administrator to register a scheduled task named "StreamerFramesServer" that runs start_server.bat at logon.

Notes:

- The scripts assume the virtualenv exists at `.framegen`. If you use a different venv, update start_server.bat / start_server.ps1 to point to the correct python.exe.
- The RIFE generator requires a CUDA-enabled PyTorch (torch + torchvision built with CUDA). Ensure `.framegen` contains those packages.
- If your environment requires additional environment variables or activation steps, update start_server.bat accordingly.

If you want, I can also add a short section showing how to create a desktop shortcut or run the server as a Windows Service.
#   f r a m e g e n v i d e o 
 
 