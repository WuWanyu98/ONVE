# ONVE

![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)
![PyTorch 2.4.1](https://img.shields.io/badge/pytorch-2.4.1-%23EE4C2C.svg)

> **Official code for the following paper**:
>
> **For Overall Nighttime Visibility: Integrate Irregular Glow Removal With Glow-Aware Enhancement**  
> Wanyu Wu, Wei Wang*, Zheng Wang, Kui Jiang, Zhengguo Li. 
> *IEEE Transactions on Circuits and Systems for Video Technology (TCSVT), 2025*
>
> **Extended from our IJCAI 2023 paper**:
> 
> **From Generation to Suppression: Towards Effective Irregular Glow Removal for
Nighttime Visibility Enhancement**

📄 [TCSVT Paper](https://ieeexplore.ieee.org/abstract/document/10685529) | 📄 [IJCAI Paper](https://ieeexplore.ieee.org/abstract/document/10685529) | 🌟 **Star this repo if you find it useful!**


---


## 📝 Pipeline
In the pursuit of Overall Nighttime Visibility Enhancement, our pipeline consists of a glow suppression module LBDN and a progressive enhancement module GRE with 3 stages. This proposed pipeline is
based on zero-shot learning without any training data. The glow is first decomposed to obtain in LBDN and refined with an APSF generator. Then the
obtained transmission D is fed into GRE for illumination enhancement via Retinex Enhancement modules REM, guided by the grayscaled glow map GM.

![flow0401](https://github.com/user-attachments/assets/d3f91e26-2df1-40d9-b250-8dd72893c55c)


---

## 🗂️ Datasets
Our method is **training-free** (no separate training phase required). The testing is performed directly on:
- [Light-effects Dataset](https://github.com/jinyeying/night-enhancement)
- [Flare7K Dataset](https://github.com/ykdai/Flare7K)

---

## 🛠️ Installation
```bash
pip install -r requirements.txt
```
---

## 🚀 Training & Testing

### Two-step Execution:

#### 1. Generate illumination hints:
```bash
python Segment.py
```

#### 2. Perform glow removal and brightness enhancement:
```bash
python Deglow.py
```
#### Key Notes:
1. We added TV loss to reduce checkerboard artifacts in glow maps, which differs from the loss setting in the paper.

2. Two adjustable parameters:

- *gamma: controls brightness enhancement intensity*

- *weight: controls the glow separation degree*

---

## 📚 Citation
If you find this repo useful, please cite our works:
```
@article{wu2024overall,
  title={For Overall Nighttime Visibility: Integrate Irregular Glow Removal with Glow-aware Enhancement},
  author={Wu, Wanyu and Wang, Wei and Wang, Zheng and Jiang, Kui and Li, Zhengguo},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2024},
  publisher={IEEE}
}

@inproceedings{wu2023generation,
  title={From generation to suppression: towards effective irregular glow removal for nighttime visibility enhancement},
  author={Wu, Wanyu and Wang, Wei and Wang, Zheng and Jiang, Kui and Xu, Xin},
  booktitle={Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence},
  pages={1533--1541},
  year={2023}
}
```

## 🙏 Acknowledgments
Our code implementation is built upon:
[Double-DIP: Unsupervised Image Decomposition via Double Deep Image Priors (CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Gandelsman_Double-DIP_Unsupervised_Image_Decomposition_via_Coupled_Deep-Image-Priors_CVPR_2019_paper.pdf)

