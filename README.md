# TFDMNet
Learning Convolutional Neural Networks in the Frequency Domain

Our code should work with TensorFlow 2.3 or later.


If you hope to use the codes of TFDMNet, please cite:

@misc{pan2024tfdmnet,
      title={TFDMNet: A Novel Network Structure Combines the Time Domain and Frequency Domain Features}, 
      author={Hengyue Pan and Yixin Chen and Zhiliang Tian and Peng Qiao and Linbo Qiao and Dongsheng Li},
      year={2024},
      eprint={2401.15949},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}


Update log 2022-07-28:

(1) We replace Leaky ReLUs using a new approximated relu layer, which is combined in the definition of approximated dropout layer. See the current arXiv edition of our paper for more details. 

(2) We update CIFAR codes, which include data preparation, network and train_test. In our CIFAR code, we implement our new memory saving technique. See the current arXiv edition of our paper for more details.

Update log 2023-09-06:

(1) We update the CNN code to the mixture model (TFDMNet)

(2) We add TFDMNet codes work with ImageNet database

Update log 2024-02-05:

(1) We update the related arxiv paper of TFDMNet
