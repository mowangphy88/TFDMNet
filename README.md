# CEMNet
Learning Convolutional Neural Networks in the Frequency Domain

If you hope to use the codes of CEMNet, please cite:

@misc{https://doi.org/10.48550/arxiv.2204.06718,
  doi = {10.48550/ARXIV.2204.06718},
  
  url = {https://arxiv.org/abs/2204.06718},
  
  author = {Pan, Hengyue and Chen, Yixin and Niu, Xin and Zhou, Wenbo and Li, Dongsheng},
  
  keywords = {Computer Vision and Pattern Recognition (cs.CV), Artificial Intelligence (cs.AI), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Learning Convolutional Neural Networks in the Frequency Domain},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {Creative Commons Attribution 4.0 International}
}


Update log 2022-07-28:
(1) We replace Leaky ReLUs using a new approximated relu layer, which is combined in the definition of approximated dropout layer. See the current arXiv edition of our paper for more details. 

(2) We update CIFAR codes, which include data preparation, network and train_test. In our CIFAR code, we implement our new memory saving technique. See the current arXiv edition of our paper for more details.
