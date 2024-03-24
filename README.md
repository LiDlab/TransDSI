<h1 align="center">TransDSI</h1>


<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#dependencies">Dependencies</a></li>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#folders">Folders</a></li>
      </ul>
    </li>
    <li>
      <a href="#usage">Usage</a>
      <ul>
        <li><a href="#dsi-prediction">DSI prediction</a></li>
        <li><a href="#dsi-binding-site-inference">DSI key sequence feature inference</a></li>
      </ul>
    </li>
    <li>
      <a href="#available-data">Available Data</a>
      <ul>
        <li><a href="#gold-standard-dataset-gsd">Gold Standard Dataset (GSD)</a></li>
        <li><a href="#benchmark-dataset">Benchmark Dataset</a></li>
        <li><a href="#predicted-dub-substrate-interaction-dataset-pdsid">Predicted DUB-Substrate Interaction Dataset (PDSID)</a></li>
      </ul>
    </li>
    <li>
      <a href="#License">License</a>
    </li>
    <li>
      <a href="#Contact">Contact</a>
    </li>
  </ol>
</details>


## About The Project
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
 ![TransDSI Architecture](results/model/Fig1.png)

TransDSI is a novel, sequence-based _ab initio_ method that leverages explainable graph neural networks and transfer learning for deubiquitinase-substrate interaction (DSI) prediction. TransDSI transfers intrinsic biological properties learned from protein sequences to predict the catalytic function of DUBs, leading to a significant improvement over state-of-the-art feature engineering methods and enabling the discovery of novel DSIs. Additionally, TransDSI features an explainable module, allowing for accurate predictions of DSIs and the identification of sequence features that suggest associations between DUBs and substrates. 



## Getting Started
To get a local copy up and running, follow these steps:

### Dependencies
TransDSI is tested to work under Python 3.7.
The required dependencies for TransDSI are  [Pytorch](https://pytorch.org/), [PyG](https://pytorch-geometric.readthedocs.io/en/latest/) and [scikit-learn](http://scikit-learn.org/).
Check environments.yml for list of needed packages.

TransDSI can run on both Windows 10 and Ubuntu 18.04 environments. We highly recommend installing and running this software on a computer with a discrete NVIDIA graphics card (models that support CUDA). If there is no discrete graphics card, the program can also run on the CPU, but it may require a longer runtime.

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/LiDlab/TransDSI.git
   ```
2. Create conda environment for TransDSI
   ```sh
   conda env create -f environment.yml
   ```
3. Based on your use, you may need to download data

   Datasets (validation and test) and features for training TransDSI are provided in [TransDSI data(~82M)](https://zenodo.org/records/10467917/files/data.tar.gz?download=1)

   Uncompress `tar.gz` file into the TransDSI directory
   ```sh
   tar -zxvf data.tar.gz -C /path/to/TransDSI
   ```
The time it takes to install the required software for TransDSI on a "normal" desktop computer is no longer than on a professional computer with a discrete graphics card. Setting up Python and the corresponding dependency packages in the Windows 10 system will not take more than 15 minutes. If you need help, please refer to the [link](https://geekflare.com/pytorch-installation-windows-and-linux/).

### Folders
./src contains the implementation for the fivefold cross-validations and independent tests of TransDSI and Baselines.

./preprocessing contains the selection of gold standard dataset and the coding of protein sequence features and similarity matrix.

./explain contains the invoking of PairExplainer, which is used to analyze the explainability of the queried DSI.

./results contains TransDSI prediction results, explainable analysis results, and trained TransDSI model.

## Usage

### DSI prediction
To predict deubiquitinase substrate interaction (DSI) use `run_DSIPredictor.py` script with the following parameters:

* `dub`             str, Uniprot ID of the queried DUB
* `candidate_sub`            str, Uniprot ID of the candidate substrate corresponding to the queried DUB
* `model_location`             str, DSIPredictor model file location

#### DEMO: obtaining the TransDSI score of [USP10-MDM2](https://www.sciencedirect.com/science/article/pii/S2211124722012761)

```sh
python run_DSIPredictor.py --dub Q14694 --candidate_sub Q00987
```
OR
```sh
python run_DSIPredictor.py -d Q14694 -s Q00987
```

#### Output:

```txt
Importing protein sequence features...
100%|███████████████████████████████████████████████████| 20398/20398 [00:10<00:00, 1993.32it/s]
Done.
Importing normalized sequence similarity matrix...
100%|█████████████████████████████████████████████| 3383863/3383863 [00:05<00:00, 598758.94it/s]
Done.
Transferred model and data to GPU
The TransDSI score of Q14694 and Q00987 is 0.9654.
```

Under normal circumstances, TransDSI typically takes around 100 seconds to predict the TransDSI score for a candidate DSI pair.
If you prefer not to utilize the GPU, you can append `--nogpu` at the end of the command.


### DSI key sequence feature inference
To investigate sequence features that suggest associations between DUBs and substrates.

use `run_PairExplainer.py` script with the following parameters:

* `feat_mask_obj`             str, The object of feature mask that will be learned (`dsi` - DSI, `dub` - DUB, `sub` - SUB)
* `dub`             str, Uniprot ID of the queried DUB
* `candidate_sub`            str, Uniprot ID of the candidate substrate corresponding to the queried DUB
* `model_location`             str, DSIPredictor model file location
* `output_location`             str, PairExplainer output file location
* `lr`             float, The learning rate to train PairExplainer
* `epochs`             int, Number of epochs to train PairExplainer
* `log`             bool, Whether or not to print the learning progress of PairExplainer

#### DEMO: obtaining the PairExplainer results of USP10-MDM2

```sh
python run_PairExplainer.py --feat_mask_obj dsi --dub Q14694 --candidate_sub Q00987 --output_location results/importance/
```
OR
```sh
python run_PairExplainer.py -obj dsi -d Q14694 -s Q00987
```

#### Output:

```txt
Importing protein sequence features...
100%|███████████████████████████████████████████████████| 20398/20398 [00:10<00:00, 1940.45it/s]
Done.
Importing normalized sequence similarity matrix...
100%|█████████████████████████████████████████████| 3383863/3383863 [00:05<00:00, 602453.09it/s]
Transferred model and data to GPU
importance this pair of DSI: 100%|████████████████████████| 10000/10000 [03:41<00:00, 45.17it/s]
The explainable result of Q14694 and Q00987 is saved in 'results/importance/Q14694_Q00987.csv'.
```

Under normal circumstances, PairExplainer takes approximately 300 seconds to predict the importance of each position on a candidate DSI pair.

If you prefer not to utilize the GPU, you can append `--nogpu` at the end of the command. However, this is not recommended as retraining PairExplainer would be necessary, which can take around 4 hours.


### Reproduction instructions for five-fold cross-validations and independent tests

If you want to replicate the five-fold cross-validation and independent testing process of TransDSI, please run the `main.py` script in the src folder.
```sh
cd src/
```
AND
```sh
python main_GSD.py && python main_GSD_TransDSI_variant.py && python main_GSD_ML.py
```

## Available Data

* #### [Gold Standard Dataset (GSD)](https://github.com/LiDlab/TransDSI/raw/master/Supplementary%20Tables/Supplementary%20Table%201.xlsx)
TransDSI has established a rigorous gold standard dataset where the positive set is sourced from [UBibroswer 2.0](http://ubibrowser.bio-it.cn/ubibrowser_v3/) and negative set is derived from [BioGRID](https://thebiogrid.org/). We divided GSD into the cross-validation dataset and the independent test dataset in chronological order.

We also provide **Gold Standard Positive Set (GSP) with inferred binding sites**, please [click](https://github.com/LiDlab/TransDSI/raw/master/Supplementary%20Tables/Supplementary%20Table%206.xlsx) to download.

* #### [Benchmark Dataset](https://github.com/LiDlab/TransDSI/tree/master/results/performance/GSD)

To ensure fair comparison, cross-validation dataset and independent test dataset are intersected with the corresponding datasets from [UbiBrowser 2.0](http://ubibrowser.bio-it.cn/ubibrowser_v3/home/download).

Click to download the [cross-validation results](https://github.com/LiDlab/TransDSI/blob/master/results/performance/GSD/GSD_crossval_prob.csv) and the [independent test results](https://github.com/LiDlab/TransDSI/blob/master/results/performance/GSD/GSD_indtest_prob.csv).

* #### [Predicted DUB-Substrate Interaction Dataset (PDSID)](https://github.com/LiDlab/TransDSI/raw/master/Supplementary%20Tables/Supplementary%20Table%204.xlsx)
TransDSI was used to performed a large-scale proteome-wide DSI scanning, resulting in a predicted DUB-substrate interaction dataset (PDSID) with 19,461 predicted interactions between 85 DUBs and 5,151 substrates.

We also provide **PDSID with inferred binding sites**, please [click](https://github.com/LiDlab/TransDSI/raw/master/Supplementary%20Tables/Supplementary%20Table%204.xlsx) to download.

## License

This project is covered under the **Apache 2.0 License**.

## Contact
Dianke Li: diankeli@foxmail.com

Yuan Liu: liuy1219@foxmail.com

Dong Li: lidongbprc@foxmail.com
