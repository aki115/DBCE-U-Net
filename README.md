# DBCE-U-Net

DBCE-U-Net is a single-object tracking algorithm proposed by Mr. Wenjun Zhou and Mr. Xiao. This repository contains the implementation of the algorithm titled **"Infrared Small Target Detection via Contrast-Enhanced Dual-Branch Network"** using PyTorch.

## Authors and Contributors
The code was implemented by:
- **Dr. Zhou** (zhouwenjun@swpu.edu.cn)
- **Mr. Xiao** 

From the Image Processing and Parallel Computing Laboratory, School of Computer Science, Southwest Petroleum University.

## Usage Notice
- If you intend to use this code to test your own algorithms, please feel free to download it.
- If you plan to use it in your publications, please inform us in advance.

Thank you for your cooperation!

*Date: Sep 5, 2024*

## Toolbox Dependency

Our model, DBCE U-Net, utilizes the [BasicIRSTD toolbox](http://github.com/XinyiYing/BasicIRSTD) for training, testing, and evaluation. This PyTorch-based open-source toolbox provides a standardized pipeline specifically designed for infrared small target detection (IRSTD) tasks, making it easy to replicate our results and compare with other IRSTD methods.

### Using the Toolbox

### Training and Testing

To train and test our model, follow the steps provided in the BasicIRSTD toolbox. 

### Datasets
* **NUAA-SIRST** &nbsp; [[download]](https://github.com/YimianDai/sirst) &nbsp; [[paper]](https://arxiv.org/pdf/2009.14530.pdf)
* **NUDT-SIRST** &nbsp; [[download]](https://github.com/YeRen123455/Infrared-Small-Target-Detection) &nbsp; [[paper]](https://ieeexplore.ieee.org/abstract/document/9864119)
* **IRSTD-1K** &nbsp; [[download dir]](https://github.com/RuiZhang97/ISNet) &nbsp; [[paper]](https://ieeexplore.ieee.org/document/9880295)

We used the NUAA-SIRST, NUDT-SIRST, IRSTD-1K for both training and test. 
To download the datasets, follow the steps provided in the BasicIRSTD toolbox. 
