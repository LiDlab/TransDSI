# DeepDSI
 ![DeepDSI Architecture](results/model/Fig1.png)

DeepDSI is a novel, sequence-based _ab initio_ method that leverages explainable graph neural networks and transfer learning for deubiquitinase-substrate interaction (DSI) prediction. DeepDSI transfers intrinsic biological properties learned from protein sequences to predict the catalytic function of DUBs, leading to a significant improvement over state-of-the-art feature engineering methods and enabling the discovery of novel DSIs. Additionally, DeepDSI features an explainable module, allowing for accurate predictions of DSIs and the identification of binding regions.

 - DeepDSI is described in the paper [“A deep learning framework for protein sequence-based deubiquitinase-substrate interaction identification”](https://github.com/Laboratory-of-biological-networks/DeepDSI) by Yuan Liu, Dianke Li, Xin Zhang, et al.


### Dependencies
DeepDSI is tested to work under Python 3.7.
The required dependencies for DeepDSI are  [Pytorch](https://pytorch.org/), [PyG](https://pytorch-geometric.readthedocs.io/en/latest/) and [scikit-learn](http://scikit-learn.org/).
To install all dependencies run:

```
conda env create -f environment.yml
```


### Folders
./src contains the implementation for the fivefold cross-validations and independent tests of DeepDSI and Baselines.

./preprocessing contains the selection of gold standard dataset and the coding of protein sequence features and similarity matrix.

./explain contains the invoking of PairExplainer, which is used to analyze the explainability of the queried DSI.

./data contains the data needed for all source code, and ./results contains the results for all source code.


### Deubiquitinase-substrate interaction (DSI) prediction
To predict DSI use `run_DSIPredictor.py` script with the following options:

* `dub`             str, Uniprot ID of the queried DUB
* `candidate_sub`            str, Uniprot ID of the candidate substrate corresponding to the queried DUB
* `model_location`             str, DSIPredictor model file location

#### Example: obtaining the DeepDSI score of [USP10-MDM2](https://www.sciencedirect.com/science/article/pii/S2211124722012761)

```
>> python run_DSIPredictor.py --dub Q14694 --candidate_sub Q00987
OR
>> python run_DSIPredictor.py -d Q14694 -s Q00987
```

#### Output:

```txt
Collect embeddings
100%|███████████████████████████████████████████████████| 20398/20398 [00:10<00:00, 1993.32it/s]
Calculate the sequence similarity matrix
100%|█████████████████████████████████████████████| 3383863/3383863 [00:05<00:00, 598758.94it/s]
Transferred model and data to GPU
The DeepDSI score of Q14694 and Q00987 is 0.9987.
```

### DSI binding site inference
To investigate the regions of the input DUB and/or candidate SUB sequence that contribute the most to the DeepDSI score

use `run_PairExplainer.py` script with the following options:

* `feat_mask_obj`             str, The object of feature mask that will be learned (`dsi` - DSI, `dub` - DUB, `sub` - SUB)
* `dub`             str, Uniprot ID of the queried DUB
* `candidate_sub`            str, Uniprot ID of the candidate substrate corresponding to the queried DUB
* `model_location`             str, DSIPredictor model file location
* `output_location`             str, PairExplainer output file location
* `lr`             float, The learning rate to train PairExplainer
* `epochs`             int, Number of epochs to train PairExplainer
* `log`             bool, Whether or not to print the learning progress of PairExplainer

#### Example: obtaining the PairExplainer results of USP10-MDM2

```
>> python run_PairExplainer.py --feat_mask_obj dsi --dub Q14694 --candidate_sub Q00987 --output_location results/importance/
OR
>> python run_PairExplainer.py -obj dsi -d Q14694 -s Q00987
```

#### Output:

```txt
Collect embeddings
100%|███████████████████████████████████████████████████| 20398/20398 [00:10<00:00, 1940.45it/s]
Calculate the sequence similarity matrix
100%|█████████████████████████████████████████████| 3383863/3383863 [00:05<00:00, 602453.09it/s]
Transferred model and data to GPU
importance this pair of DSI: 100%|████████████████████████| 10000/10000 [06:49<00:00, 24.43it/s]
The explainable result of Q14694 and Q00987 is saved in 'results/importance/Q14694_Q00987.csv'.
```
