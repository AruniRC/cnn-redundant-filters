## Redundant filters in CNNs

### Setup

* (Install Anaconda)[https://conda.io/docs/user-guide/install/linux.html] if not already installed in the system.
* Create an Anaconda environment: `conda create -n cnn-duplicates python=2.7` and activate it: `source activate cnn-duplicates`.
* Install PyTorch and TorchVision inside the Anaconda environment. First add a channel to conda: `conda config --add channels soumith`. Then install: `conda install pytorch torchvision cuda80 -c soumith`.
* Setup from cloned repo: 
    * `git clone git@github.com:AruniRC/cnn-redundant-filters.git`
    * Inside the local folder for this repo, add the submodule forked from [WideResNet-pytorch](https://github.com/xternalz/WideResNet-pytorch):  `git submodule add https://github.com/AruniRC/WideResNet-pytorch WideResNet-pytorch` 
* [TODO] Dependencies:
    * `pip install tensorboard_logger`
    * `conda install -c conda-forge tensorboard`
    * 



* Install the dependencies using conda: `conda install scipy Pillow tqdm scikit-learn scikit-image numpy matplotlib ipython pyyaml`.


### Usage

#### Train WideResNet

Enter the `WideResNet-pytorch` module folder and run: `python train.py --dataset cifar10 --layers 28 --widen-factor 10 --tensorboard`.

You may need to modify the lines that specify numbebr of GPUs to use in the `train.py` script. By default it trains for 200 epochs on CIFAR-10 and saves logs for TensorBoard. 



#### Demo: reduce LeNet CNN

A demo script showing how to specify layers to be reduced/pruned in a LeNet-style toy CNN is given in `demo_reduce_cifar_cnn.py`. The top of the script defines several variables where a user can specify the experiment settings along with explanatory comments.  This removes a specified number of filters using either the norm-based or the cosine-similarity-based criterion, and reports the test accuracy of the network before and after the pruning operation. 








