# KD

This repo covers the implementation of the following paper:

"Data Tells the Truth: A Knowledge Distillation Method
for Genomic Survival Analysis by Handling Censoring"

Project home page: https://datatellstruth.github.io/.


## Requeirements

Python 3.7.11

Pytorch 1.13.1

numpy 1.21.2

scikit-learn 1.0

tqdm 4.62.4

```
python setup.py install
```

## Data preparation

All data used in our work are publicly available.  You can download the TCGA data from :https://www.synapse.org/#!Synapse:syn4976369.2. Here, we provide processed files in `data/raw_data`. Data files required for running can be obtained through processing. For instance, you can process `data/raw_data/Oncogene_5.txt` to obtain the {cancer_type}_gene5.csv. We provide samples in `data/sample_data`. The folder stucture as follow:

````
```
data
  |————gene
  |       └——————ACC.csv
  |       └——————ACC_Clean.csv
  |       └——————PAAD.csv
  |       └——————PAAD_Clean.csv
  |       └——————...
  |————location
  |       └——————ACC_gene5.csv
  |       └——————HNSC_gene5.csv
  |       └——————PAAD_gene5.csv
  |       └——————LUAD_gene5.csv
  |       └——————...
  ```
````

## Usage

Path to the data used for running should save in `args.csv_folder` as csv files, you can use the other data formats to match the dataset in your machine or re-write the `Cancer` and `Cancer_fold` class in the `dataset.py`.

We have encapsulated the methods into a Python library, which includes two trainers: `KD_Trainer` and `KDKA_Trainer`. Additionally, there is an evaluation function provided for assessment. Below, we provide an example demonstrating how to use these methods:

```python
from KD.experiment import KD_Trainer, KDKA_Trainer
# For more details about parameters, please refer to the github repository.

#train = KDKA_Trainer(cancer_type='ACC', folder='data/cancer_gene/', resume='', csv_folder='data/', round=25, batch_size=10, output_folder='pth', lr=0.1, dim=5796, seed=42)
train = KD_Trainer(cancer_type='ACC', folder='data/cancer_gene/', resume='', csv_folder='data/', round=25, batch_size=10, output_folder='pth', lr=0.1, dim=5796, seed=42)
```

We have also provided a runnable notebook file for ease of learning and utilization.
