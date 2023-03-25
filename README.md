# KDKA

This repo covers the implementation of the following paper:

"A Unifified Method of Knowledge Distillation and Knowledge Abduction for Genomic Survival Analysis"



## Requeirements

Python 3.7.11

Pytorch 1.13.1

numpy 1.21.2

scikit-learn 1.0

tqdm 4.62.4

```
pip install -r requirements.txt
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

Path to the data used for running should save in `args.folder` as pickles, you can use the other data formats to match the dataset in your machine or re-write the *Cancer* and *Cancer_folder* class in the *datasets.py*.

### Train

Take PAAD cancer as an example,  run:

`python main.py --folder <data path> --mode <all/few-shot> --cancer_type PAAD`

### Test

Take PAAD cancer as an example,  run:

`python test.py --folder <data path> --mode <all/few-shot> --cancer_type PAAD --resume <pth path>`



