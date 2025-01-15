# IFBlend

This repository covers the "Towards Ambient Lighting Normalization" paper, accepted for publication at ECCV 2024. 
Further materials covering our work will be available through this repo. Follow the current README for more resources. 

# Checkpoints

IFBlend checkpoints trained on AMBIENT6K are available [here](https://drive.google.com/file/d/1nNY1vF7mwVRWgTtdhJRmF5s9yGCAMZfm/view?usp=sharing).

# Installing
 * Clone the current repository 

    ```bash 
   git clone https://github.com/fvasluianu97/IFBlend.git
    ```
 * Download the <code>checkpoints.zip</code> from [here](https://drive.google.com/file/d/1nNY1vF7mwVRWgTtdhJRmF5s9yGCAMZfm/view?usp=sharing) and unzip to the repository root directory. 
 * Download the <code>weights.zip</code> from [here](https://drive.google.com/file/d/1rwv2G8tAboGEzEsczMMiSMzxCA1to3f7/view?usp=sharing) and unzip in the root directory of the repository. 
 * Activate your Python virtual environment.
 * Install the packages mentioned in <code>requirements.txt</code>.
 * Test your IFBlend checkpoint:
```bash
   python eval.py --data_src ./data/AMBIENT6K --ckp_dir ./checkpoints --res_dir ./final-results --load_from IFBlend_ambient6k
```

# Data
In the next section you can find the URLs for the training, testing, nad benchmark splits of the AMBIENT6K dataset. Download the available resources to a file tree similar to teh following structure:

```
.
├── checkpoints
│   └── IFBlend_ambient6k
│       └── best
│           └── checkpoint.pt
├── data
│   └── AMBIENT6K
│       ├── Benchmarck
│       ├── Test
│       │   ├── 0_gt.png
│       │   ├── 0_in.png
│       │   ├── 1_gt.png
│       │   └── 1_in.png
│       └── Train
│           ├── 0_gt.png
│           ├── 0_in.png
│           ├── 1_gt.png
│           └── 1_in.png
├── dataloader.py
├── dconv_model.py
├── eval.py
├── final-results
├── .gitignore
├── ifblend.py
├── laynorm.py
├── loaded_models
├── loss.py
├── metrics.py
├── model_convnext.py
├── perceptual_loss.py
├── README.md
├── refinement.py
├── requirements.txt
├── train.py
├── unet.py
├── utils_model.py
├── utils.py
└── weights
    └── convnext_xlarge_22k_1k_384_ema.pth

```


# AMBIENT6K
The AMBIENT6K dataset is designed to drive the research in the field of Lightning Normalization, as a collection of 6000 
high resolution images representing images affected by non-homogeneous lighting distribution. For each affected image we propose
a ground-truth image representing the same scene, but under near-perfect lighting conditions. The ground-truth images are 
representations of canonical lighting, characteristic to a professional photography setup. 

The images are shot with a Canon R6 mk2 camera, at 24 MP. The resolution of the images used in our experiments had to be limited 
due to the computing resources limitations. The data is available in Canon CR3 RAW image format, so using software like Adobe Lightroom
would enable exporting it at 24 MP resolution. 
For reproducibility, we will upload the version used in our ablations here also. 

* Training data: [RGB inp. img. ](https://drive.google.com/drive/folders/13O-ssekl9IrylQW9G9Bi-DMgHAtUbA2g?usp=sharing) |
[RAW inp. img. ](https://drive.google.com/drive/folders/1bYHpTTnQSYuXTUWUF9YUsRQReoX1SuTf?usp=sharing) |
[RGB gt. img. ](https://drive.google.com/drive/folders/1nl3MtA33Ze0rNj57rDFkXBIkGaNi5bFD?usp=sharing) |
[RAW gt. img. ](https://drive.google.com/drive/folders/18hfyq6bpycUVMJ5RAMuZtYeEw80y26SJ?usp=sharing) |
[scene metadata](https://drive.google.com/file/d/1fEHO-ZyMYJLM0NBNwSgNtwpkl4_ucKoR/view?usp=sharing) |
[object metadata](https://drive.google.com/file/d/1TaZLcR3pYXHGCAcSVm0gYwjXmPUBJM1P/view?usp=sharing)

* Testing data: [RGB inp. img. ](https://drive.google.com/drive/folders/14FFLoIcrFI6Rnykb3-UWJL08dF86qfNO?usp=sharing) |
[RAW  inp. img. ](https://drive.google.com/drive/folders/15Q9svT0dnAMcwz3CbUmFe8ZNHVYubzLd?usp=sharing) |
[RGB gt. img. ](https://drive.google.com/drive/folders/1AfEYAJZAh-yiV61Cs1fnxlbu4psZLBvc?usp=sharing) |
[RAW gt. img. ](https://drive.google.com/drive/folders/12c_y9-vQHXt5jnZ5QzjN56H4NfZdmXvZ?usp=sharing) |
[scene metadata](https://drive.google.com/file/d/17h3D8eGOTvnor9nQ1-fcBCIqHlTFzKP6/view?usp=sharing) |
[object metadata](https://drive.google.com/file/d/1Q6x8IieZh4koQfmQoYgMGL0brbR00pMo/view?usp=sharing)
* Benchmark data: [RGB img.](https://drive.google.com/drive/folders/1knkarmPV5d2yg7WJFXCGIjdqfVuOpGEn?usp=sharing) |
[RAW img.](https://drive.google.com/drive/folders/1AwZ55UW9Ys_CbokF_6gf9JgwumYSziNb?usp=sharing).  

# Data Structure
The RAW images are organized per scene. Each scene is noted in the metadata and each scene corresponds to a ground truth RAW image. 
In the RGB data, each input image was paired to the corresponding ground-truth image.  

# Acknowledgements
The following repositories represented valuable resources in our work: 
* https://github.com/megvii-research/NAFNet.git
* https://github.com/fvasluianu97/WSRD-DNSR.git
* https://github.com/liuh127/NTIRE-2021-Dehazing-DWGAN.git

We thank the authors for sharing their work!



