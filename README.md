<div align="center">

<h1>APasco: High Fidelity Audio-Driven Portrait Animitation based on Audio-Lip Multi-Head Cross-Attention and 3D Dense Geometric Prior </h1>

<div>
    Jinhan Xie<sup>1</sup>&emsp;
    Kanglin Liu<sup>2</sup>&emsp;
    Zhenyu Bao<sup>1</sup>&emsp;
    Qing Li<sup>2, *</sup>
</div>

<div>
    <sup>1</sup>Peking University&emsp;
    <sup>2</sup>Pengcheng Laboratory
</div>

<div>
    <sup>*</sup>corresponding author
</div>

### [Paper](https://ebooks.iospress.nl/volumearticle/76129?_gl=1*10k29o5*_up*MQ..*_ga*MTYwNjQ4ODkyMC4xNzYzNjI3MTA2*_ga_6N3Q0141SM*czE3NjM2MjcxMDYkbzEkZzEkdDE3NjM2MjcxMzAkajM2JGwwJGgw) | [Project](https://github.com/pcl3dv/APasco?tab=readme-ov-file)  | Code 

</div>

![image](assets/construct3.png)

# Abstract
<div>
    Audio-driven portrait animation has achieved significant advances propelled by the development of diffusion models.
Despite remarkable improvements in driving capability and temporal consistency, diffusion model-based methods still suffer from audio-lip misalignment and facial detail loss.
To address them, we present a novel stable diffusion-based approach by conditioning on aligned audio-lip features and 3D dense sequential geometry features. Specifically, we enhance phoneme-lip synchronization by coupling fine-grained local lip features with corresponding audio details with the designed Audio-Lip multi-head Cross-Attention module.
To improve the facial local details,  we derive 3D dense sequential geometry features from  3D dense geometric prior via the developed Mesh Spatio-Temporal Encoder.
Extensive experiments on public benchmarks demonstrate that APasco achieves superior performance in both visual quality and lip-sync accuracy compared to existing approaches.
</div>

<br>

# Results

[https://github.com/user-attachments/assets/09861fcc-18b0-4eae-9d7d-f298f3bdc181](https://github.com/pcl3dv/APasco/blob/main/assets/our.mp4)

[https://github.com/user-attachments/assets/0cdd3f0c-efbf-415b-be80-252fb4a5f121](https://github.com/pcl3dv/APasco/blob/main/assets/our1.mp4)

<br>

## Installation

- System requirement: Ubuntu 20.04/Ubuntu 22.04, Cuda 12.1
- Tested GPUs: A100

Create conda environment:

```bash
  conda create -n apasco python=3.10
  conda activate apasco
```

Install packages with `pip`

```bash
  pip install -r requirements.txt
  pip install .
```

Install ffmpeg :
```bash
  apt-get install ffmpeg
```
### Download Pretrained Models
- [audio_separator](https://huggingface.co/huangjackson/Kim_Vocal_2): Kim\_Vocal\_2 MDX-Net vocal removal model. (_Thanks to [KimberleyJensen](https://github.com/KimberleyJensen)_)
- [insightface](https://github.com/deepinsight/insightface/tree/master/python-package#model-zoo): 2D and 3D Face Analysis placed into `pretrained_models/face_analysis/models/`. (_Thanks to deepinsight_)
- [face landmarker](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task): Face detection & mesh model from [mediapipe](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker#models) placed into `pretrained_models/face_analysis/models`.
- [motion module](https://github.com/guoyww/AnimateDiff/blob/main/README.md#202309-animatediff-v2): motion module from [AnimateDiff](https://github.com/guoyww/AnimateDiff). (_Thanks to [guoyww](https://github.com/guoyww)_).
- [sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse): Weights are intended to be used with the diffusers library. (_Thanks to [stablilityai](https://huggingface.co/stabilityai)_)
- [StableDiffusion V1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5): Initialized and fine-tuned from Stable-Diffusion-v1-2. (_Thanks to [runwayml](https://huggingface.co/runwayml)_)
- [wav2vec](https://huggingface.co/facebook/wav2vec2-base-960h): wav audio to vector model from [Facebook](https://huggingface.co/facebook/wav2vec2-base-960h).

###  Run Inference

```bash
python scripts/inference.py --source_image examples/reference_images/test.jpg --driving_audio examples/driving_audios/test.wav
```

## Citation

Cite as below if you find this repository is helpful to your project:
```
@inproceedings{apasco2025,
  title={APasco: High Fidelity Audio-Driven Portrait Animitation based on Audio-Lip Multi-Head Cross-Attention and 3D Dense Geometric Prior},
  author={Xie, Jinhan and Liu, Kanglin and Bao, Zhenyu and Li, Qing},
  booktitle={ECAI},
  pages={3258--3265},
  year={2025}
}
```
