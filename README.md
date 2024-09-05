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

To train and test our model, follow the steps provided in the BasicIRSTD toolbox. The toolbox includes comprehensive instructions and scripts for data loading, preprocessing, training, and evaluation. By using the toolbox, you can ensure that the training and testing processes align with standardized IRSTD benchmarks.

## Table of Contents
- [Environment Setup](#environment-setup)
- [Testing the Tracker](#testing-the-tracker)
- [Evaluating the Tracker](#evaluating-the-tracker)
- [Training](#training)
- [Acknowledgement](#acknowledgement)

## Environment Setup
This code has been tested with the following environment:
- **OS:** Ubuntu 22.04
- **Python:** 3.8
- **PyTorch:** 1.10.0
- **CUDA:** 11.3

To set up the environment, please install the required libraries:

```bash
pip install -r requirements.txt
