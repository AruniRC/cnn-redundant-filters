## Redundant filters in CNNs

### Setup

* (Install Anaconda)[https://conda.io/docs/user-guide/install/linux.html] if not already installed in the system.
* Create an Anaconda environment: `conda create -n cnn-duplicates python=2.7` and activate it: `source activate cnn-duplicates`.
* Install PyTorch and TorchVision inside the Anaconda environment. First add a channel to conda: `conda config --add channels soumith`. Then install: `conda install pytorch torchvision cuda80 -c soumith`.
* Setup from cloned repo: 
    * `git clone git@github.com:AruniRC/cnn-redundant-filters.git`
    * Inside the local folder for this repo, add the submodule forked from [WideResNet-pytorch](https://github.com/xternalz/WideResNet-pytorch):  `git submodule add https://github.com/AruniRC/WideResNet-pytorch WideResNet-pytorch` 
* Dependencies:
    * `pip install tensorboard_logger`
    * `conda install -c conda-forge tensorboard`
    * 



* Install the dependencies using conda: `conda install scipy Pillow tqdm scikit-learn scikit-image numpy matplotlib ipython pyyaml`.


### Usage



