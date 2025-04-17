# DBCE-U-Net

DBCE-U-Net is a infrared small target detection algorithm proposed by Dr. Wenjun Zhou and Mr. Xiao. This repository contains the implementation of the algorithm from the paper **"Infrared Small Target Detection via Contrast-Enhanced Dual-Branch Network"** using PyTorch.

## Authors and Contributors

This code was implemented by:

- Dr. Wenjun Zhou (Email: zhouwenjun@swpu.edu.cn)
- Mr. Xiao

From the Image Processing and Parallel Computing Laboratory, School of Computer Science, Southwest Petroleum University.

## Usage Notice

- Feel free to download and use this code for testing your algorithms.
- If you use this code in your publications, please inform us in advance.

Thank you for your cooperation!

Date: Sep 5, 2024

## Dependencies

Our model, DBCE-U-Net, utilizes the [BasicIRSTD toolbox](http://github.com/XinyiYing/BasicIRSTD) for training, testing, and evaluation. This open-source toolbox, based on PyTorch, provides a standardized pipeline specifically designed for infrared small target detection (IRSTD) tasks, facilitating easy replication of our results and comparison with other IRSTD methods.

### Using the Toolbox

Please refer to the instructions in the BasicIRSTD toolbox for training, testing, and evaluation of our model.

### Datasets

We used the following datasets for both training and testing:

1. **NUAA-SIRST**

   - [Download](https://github.com/YimianDai/sirst)
   - [Paper](https://arxiv.org/pdf/2009.14530.pdf)
2. **NUDT-SIRST**

   - [Download](https://github.com/YeRen123455/Infrared-Small-Target-Detection)
   - [Paper](https://ieeexplore.ieee.org/abstract/document/9864119)
3. **IRSTD-1K**

   - [Download](https://github.com/RuiZhang97/ISNet)
   - [Paper](https://ieeexplore.ieee.org/document/9880295)

For detailed instructions on how to use these datasets, please refer to the BasicIRSTD toolbox documentation.

## Acknowledgement

The code is implemented based on [BasicIRSTD toolbox](http://github.com/XinyiYing/BasicIRSTD). We would like to express our sincere thanks to the contributors.
