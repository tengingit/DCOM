## DCOM

This is an implementation of the paper
Teng Huang, Bin-Bin Jia, Min-Ling Zhang. Towards Escaping from Class Dependency Modeling for Multi-Dimensional Classification. In: Proceedings of the 42rd International Conference on Machine Learning (ICML'25), Vancouver, Canada.

Github link: https://github.com/tengingit/DCOM-ICML-25
***

## Requirements

- Python == 3.9.18
- Pytorch == 1.12.1
- numpy == 1.26.0
***

### Datasets

All tabular data sets can be downloaded from https://palm.seu.edu.cn/zhangml/Resources.htm#MDC_data.
To get access to image data sets adopted , please kindly refer to detailed descriptions in Appendix B of the paper.

### Train and Test

For example, to perform 10-fold cross validation on tabular datasets such as *Song*:

```
python DAEMDC_ci.py -dataset Song
```

To perform DCOM on image datasets such as *DeepFashion*:

```
python DAEMDC_ci_img_pretrained.py -dataset DeepFashion
```

We keep the training log on **'logs'** directory and testing results on **'results'** directory.






