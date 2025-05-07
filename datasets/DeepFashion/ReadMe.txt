Tables of Contents
==================
- Descriptions
- Original Sources
- References
- Data Format
- Preprocessing Notes
- Statistical Information


Descriptions
============
The DeepFashion dataset deals with the fine version of Category and Attribute Prediction task in Large-scale Fashion (DeepFashion) Database. Category and Attribute Prediction Benchmark evaluates the performance of clothing category and attribute prediction. This is a large subset of DeepFashion, containing massive descriptive clothing categories and attributes in the wild.


Original Sources
================
1. Large-scale Fashion (DeepFashion) Database: https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html


References
==========
1. The original DeepFashion data set is presented in the following paper:
Liu, Ziwei and Luo, Ping and Qiu, Shi and Wang, Xiaogang and Tang, Xiaoou. "DeepFashion: Powering Robust Clothes Recognition and Retrieval with Rich Annotations." In: Proceedings of the 27th IEEE Conference on Computer Vision and Pattern Recognition (CVPR'2016), Las Vegas, USA, 2016, 1096-1104.


Data Format 
====================
"images": 20000 images from Large-scale Fashion (DeepFashion) Database.
"image_name": A matlab list contains names of image files (img/xxx.jpg).
"targets": A 20000x6 table where the six columns corresponds to dimension "texture", "fabric", "shape", "part", "style" and "elasticity" respectively. Each row corresponds to an image in which the order of image is in accordance with the order of names in 'image_name.mat".
"list_attr_cloth": First Row: number of attributes; Second Row: entry names; Rest of the Rows: <label name> <label type>.  In label type, "1" represents texture-related labels, "2" represents fabric-related labels, "3" represents shape-related labels, "4" represents part-related labels, "5" represents style-related labels, "6" represents elasticity-related labels.
"list_attr_img": First Row: number of images; Second Row: entry names; Rest of the Rows: <image name> <labels>.


Preprocessing Notes
===================
1. We only adopted the fine version of DeepFashion dataset given its multi-dimensional characteristic. The "elasticity" dimension is not mentioned in the original paper. However, in the "list_attr_cloth.txt" in the "Anno_fine" folder of Category and Attribute Prediction Benchmark of DeepFashion, we found 3 labels "tight", "loose" and "conventional" which are denoted by "6". We named the sixth dimension as "elasticity" for the moment.
2. For the labels (refered to as "attribute" in the original paper), raw data used one-hot encoding to represent the relevance of labels with images. We transformed one-hot encoding to numeral labels according to the position of "1" (starting from 0). For the 26 labels in "list_attr_cloth.txt", take one-hot encoding "0 0 1 0 0 0 0 0 0 1 0 0 1 0 0 0 1 0 0 1 0 0 0 0 0 1" (please refer to the third row in "list_attr_img.txt") as an example, the transformed label vector is [2, 2, 2, 3, 2, 2] (please refer to the first row in "targets.csv").


Statistical Information
===================
num_dim: 6
label_per_dim: {0: [0, 1, 2, 3, 4, 5, 6], 1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2, 3], 4: [0, 1, 2, 3, 4, 5], 5: [0, 1, 2]}
num_per_dim: [7 3 3 4 6 3]
number of labels in each dimension:
0: {0: 3402, 1: 3346, 2: 1347, 3: 1667, 4: 414, 5: 9482, 6: 342};
1: {0: 6117, 1: 3474, 2: 10409}, 2: {0: 2029, 1: 3834, 2: 14137};
3: {0: 8086, 1: 3458, 2: 61, 3: 8395};
4: {0: 1194, 1: 2885, 2: 13604, 3: 406, 4: 189, 5: 1722};
5: {0: 2891, 1: 1069, 2: 16040}