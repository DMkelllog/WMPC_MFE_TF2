# Wafer map pattern classification using MFE

Wafer map defect pattern classification using manual feature extraction

## Methodology

### Manual Feature extraction based model

![](https://github.com/DMkelllog/WMPC_MFE/blob/main/MFE%20flow.PNG?raw=true)

* Input:    handcrafted features from a wafer map
  * Density (13), Geometry (6), Radon (20)
* Output: predicted class
* Model:  ML classifier (FNN)
* Manual feature extraction code from https://www.kaggle.com/ashishpatel26/wm-811k-wafermap

## Data

* WM811K
  * 811457 wafer maps collected from 46393 lots in real-world fabrication

  * 172950 wafers were labeled by domain experts.

  * 9 defect classes (Center, Donut, Edge-ring, Edge-local, Local, Random, Near-full, Scratch, None)

    

  * provided by MIR Lab (http://mirlab.org/dataset/public/)

  * .pkl file downloaded from Kaggle dataset (https://www.kaggle.com/qingyi/wm811k-wafer-map)

  * directory: /data/LSWMD.pkl

## Dependencies

* Python
* Pandas
* Tensorflow
* Scikit-learn
* Scikit-image

## References

* WM-811K(LSWMD). National Taiwan University Department of Computer Science Multimedia Information Retrieval LAB http://mirlab.org/dataSet/public/
* Wu, M. J., Jang, J. S. R., & Chen, J. L. (2014). Wafer map failure pattern recognition and similarity ranking for large-scale data sets. IEEE Transactions on Semiconductor Manufacturing, 28(1), 1-12.
* Fan, M., Wang, Q., & van der Waal, B. (2016, October). Wafer defect patterns recognition based on OPTICS and multi-label classification. In 2016 IEEE Advanced Information Management, Communicates, Electronic and Automation Control Conference (IMCEC) (pp. 912-915). IEEE.
* Saqlain, M., Jargalsaikhan, B., & Lee, J. Y. (2019). A voting ensemble classifier for wafer map defect patterns identification in semiconductor manufacturing. IEEE Transactions on Semiconductor Manufacturing, 32(2), 171-182.
