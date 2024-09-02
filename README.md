PRITrans
===

Protein-RNA interactions are essential for numerous cellular processes, and missense mutations in RNA-binding proteins can disrupt these interactions, contributing to disease pathogenesis. We introduced PRITrans, a Transformer-based deep learning framework to predict the effects of missense mutations on protein-RNA interactions. Compared to existing methods, PRITrans shows significant performance improvements by integrating ESM-2 and ProtTrans features and utilizing both Transformer and multi-scale convolution modules. This approach enhances feature extraction and model architecture, leading to better prediction accuracy and stability. Furthermore, future research should focus on optimizing the model architecture for higher accuracy. Expanding PRITrans to other bioinformatics applications and incorporating additional biological data will improve its generalization and precision. Thus, PRITrans has the potential to significantly impact biomedical research, supporting both scientific and practical applications.

Install Dependencies
===
  h5py==2.10.0\
  matplotlib==3.5.0\
  Keras==2.3.1\
  scikit-learn==1.2.2\
  scipy==1.4.1\
  seaborn==0.13.2\
  tensorboard==2.14.0\
  tensorflow==2.3.0\
  numpy==1.18.5\
  pandas==1.1.5\
  ppython==3.8

Run
===

We provide the regression model and classification model files used in this paper, namely codes/model_regression.py and codes/model_classify.py, respectively.\
We provide two essential component modules from the paper, namely codes/Encoder.py and codes/Mutil_scale_prot.py, respectively.\
To train your own data, use the codes/train_kfold_regression.py and codes/train_kfold_classify.py files, respectively.\
To validate the model's performance on the independent test set, please use the codes/test_indep_regression.py file.

Contact
===

If you are interested in our work, OR, if you have any suggestions/questions about our work, PLEASE contact with us. E-mail: 231210701110@stu.just.edu.cn



