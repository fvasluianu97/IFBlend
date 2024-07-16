# IFBlend


This repository covers the "Towards Ambient Lighting Normalization" paper, accepted for publication at ECCV 2024. 
Further materials covering our work will be available through this repo. Follow the current README for more resources. 

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