# ONVE

![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)

> **Official code for TCSVT 2025**:  
> **For Overall Nighttime Visibility: Integrate Irregular Glow Removal With Glow-Aware Enhancement**  
> [Wanyu Wu], [Wei Wang]*, [Zheng Wang], [Kui Jiang], [Zhengguo Li]. 
> *IEEE Transactions on Circuits and Systems for Video Technology (TCSVT), 2025*  
> **Extended from our IJCAI 2023 paper**: [From Generation to Suppression: Towards Effective Irregular Glow Removal for
Nighttime Visibility Enhancement]  

📄 [TCSVT Paper](https://ieeexplore.ieee.org/abstract/document/10685529) | 📄 [IJCAI Paper](https://ieeexplore.ieee.org/abstract/document/10685529) | 🌟 **Star this repo if you find it useful!**

---

## 🗂️ Datasets
Our method is **training-free** (no separate training phase required). The testing is performed directly on:
- [Sharma Dataset](http://cvlab.postech.ac.kr/research/illumination_enhancement/)
- [Flare7K Dataset](https://github.com/ykdai/Flare7K)

---

## 🛠️ Installation
```bash
conda create -n deglow python=3.8
conda activate deglow
pip install -r requirements.txt
```
---

## 🚀 Training & Testing

### Two-step Execution:

#### 1. Generate illumination hints:
```bash
python segment.py
```

#### 2. Perform glow removal and enhancement:
```bash
python Deglow.py
```
#### Key Notes:
We added TV loss to reduce checkerboard artifacts in glow maps

Two adjustable parameters:

gamma: controls brightness enhancement intensity

weight: controls the glow separation degree

---

## 📚 Citation
If you use this code, please cite both our works:

bibtex
@article{your2025tcsvt,
  title={Your TCSVT 2025 Paper Title},
  author={Name, Your and Author, Co},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2025}
}

@inproceedings{your2023ijcai,
  title={Your IJCAI 2023 Paper Title},
  author={Name, Your and Author, Co},
  booktitle={Proceedings of the International Joint Conference on Artificial Intelligence},
  year={2023}
}

## 🙏 Acknowledgments
This implementation is built upon:
Double-DIP: Unsupervised Image Decomposition via Double Deep Image Priors (CVPR 202X)

