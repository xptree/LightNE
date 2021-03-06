# LightNE

Code for SIGMOD 2021 submission *LightNE: A Lightweight Graph Processing System for Network Embedding*.

LightNE algorithm is in the LightNE directory.


## Dependencies Installation

### Install g++
The code is compiled and run with g++ 6.5.0 and 5.4.0 (any supporting c++17 should work in theory).

### Install Boost
In the spectral propagation step, we need modified Bessel functions of the first kind. Boost provides such functions. So we need to install Boost.
```
sudo apt-get install libboost-dev
```

### Install Intel MKL

There are two ways to install Intel MKL. 

* The first way is to install with Anaconda (recomended)
```
conda create -n lightne python=3.7 # first create a new python env
conda activate lightne # activate the new created env
conda install mkl -c intel --no-update-deps
conda install mkl-devel
```

* The second way is to download directly from Intel. Please follow
```
https://software.intel.com/en-us/mkl/choose-download/linux
```
You will download something named `parallel_studio_xe_2019_update4_cluster_edition_online.tgz` and the installation script will install intel mkl (by default) at `/opt/intel`.


### Install other necessary dependencies

The preprocessing script (which translate .mat or .edgelist graph to AdjacencyGraph format) and the evaluation script requires the following python libs:
```
pip install sklearn pandas scipy
```

## Compile
To compile Ligne, you need to edit `Makefile` a little, indicating the directories of your Intel MKL. 

If you install intel mkl directly, then you need to set something like:
```
INCLUDE_DIRS = -I../ -I/opt/intel/mkl/include
LINK_DIRS = -L"/opt/intel/mkl/lib/intel64"
```
Otherwise, if install with Anaconda, then you need to set something like:
```
INCLUDE_DIRS = -I../ -I"/home/XXX/anaconda3/envs/lightne/include"
LINK_DIRS = -L"/home/XXX/anaconda3/envs/lightne/lib"
```

Then run `make` to compile.

To clean the compiled file, run `make clean`.

## Run


### BlogCatalog

I have uploaded BlogCatalog dataset and this git repo (at `data_bin/blogcatalog.mat`). 

If your intel mkl is installed directly, you need something like:
```
export LD_LIBRARY_PATH=/opt/intel/mkl/lib/intel64
```
If installed with Anaconda, then you need something like:
```
export LD_LIBRARY_PATH=/home/XXX/anaconda3/envs/lightne/lib
```

Then you can run `blog_lightne.sh`. The running log is stored at `blog_lightne.log`


### Datasets from NetSMF paper

Download and unzip datasets used in NetSMF paper
```
cd data_bin
wget https://sampledbsql1backup.blob.core.windows.net/www19netsmf/datasets.zip
unzip datasets.zip
```

unzip will give you the following files:
```
Archive:  datasets.zip
  inflating: blogcatalog.mat
  inflating: flickr.mat
  inflating: mag.edge
  inflating: mag.label.npz
  inflating: MicrosoftResearchDataLicenseAgreement.pdf
  inflating: ppi.mat
  inflating: Readme.txt
  inflating: youtube.mat
```

Besides BlogCatalog, You can run `youtube_lightne.sh`, `mag_lightne.sh` for each dataset.

### Very Large Graphs

* ClueWeb graph can be downloaded from [here](http://law.di.unimi.it/webdata/clueweb12/).
* Hyperlink2014 graph can be downloaded from [here](http://webdatacommons.org/hyperlinkgraph/2014-04/download.html).

