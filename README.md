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

### [Paper](https://ebooks.iospress.nl/volumearticle/76129?_gl=1*10k29o5*_up*MQ..*_ga*MTYwNjQ4ODkyMC4xNzYzNjI3MTA2*_ga_6N3Q0141SM*czE3NjM2MjcxMDYkbzEkZzEkdDE3NjM2MjcxMzAkajM2JGwwJGgw) | [Project](https://zhenybao.github.io/MGStream/)  | Code 

</div>

![image](assets/teaser_figure.png)

# Abstract
<div>
    3D Gaussian Splatting (3DGS) has gained significant attention in streamable dynamic novel view synthesis (DNVS) for its photorealistic rendering capability and computational efficiency. Despite much progress in improving rendering quality and optimization strategies, 3DGS-based streamable dynamic scene reconstruction still suffers from flickering artifacts and storage inefficiency, and struggles to model the emerging objects. To tackle this, we introduce MGStream which employs the motion-related 3D Gaussians (3DGs) to reconstruct the dynamic and the vanilla 3DGs for the static. The motion-related 3DGs are implemented according to the motion mask and the clustering-based convex hull algorithm. The rigid deformation is applied to the motion-related 3DGs for modeling the dynamic, and the attention-based optimization on the motion-related 3DGs enables the reconstruction of the emerging objects. As the deformation and optimization are only conducted on the motion-related 3DGs, MGStream avoids flickering artifacts and improves the storage efficiency. Extensive experiments on real-world datasets N3DV and MeetRoom demonstrate that MGStream surpasses existing streaming 3DGS-based approaches in terms of rendering quality, training/storage efficiency and temporal consistency.
</div>

<br>

# Comparasion and Free view synthesis

https://github.com/user-attachments/assets/09861fcc-18b0-4eae-9d7d-f298f3bdc181

https://github.com/user-attachments/assets/0cdd3f0c-efbf-415b-be80-252fb4a5f121

<br>

## Citation

Cite as below if you find this repository is helpful to your project:
```
@misc{bao2025mgstreammotionaware3dgaussian,
      title={MGStream: Motion-aware 3D Gaussian for Streamable Dynamic Scene Reconstruction}, 
      author={Zhenyu Bao and Qing Li and Guibiao Liao and Zhongyuan Zhao and Kanglin Liu},
      year={2025},
      eprint={2505.13839},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.13839}, 
}
```
