# GORetriever web demo

web demo for GORetriever, cloned from https://github.com/ZhuLab-Fudan/GORetriever/

# Installing

Get python 3.8

```
python3.8 -m venv venv
source venv/bin/activate
pip install -r requirements.txt --no-deps
```

Download cross_encoder from https://drive.google.com/file/d/11W51FnM62Z79qGPkuZHRzAv6Bx_L1Mah/view?usp=sharing into folder **cross_model**

# Running


```
python predict.py \
  --task [branch] \
  --pro [k] \
  --gpu [n] \
  --file [file]
```

branch: bp, mf, cc

pro: number of retrieved proteins, set k = 3 for MFO and BPO and k = 2 for CCO 

gpu: cuda number, default as 0

file: filename for test set, eg: ./data/test.txt

eg:
```
python predict.py --task bp --pro 3 --gpu 0
```

# Start your own experiments

If you want to start your new experiment, delete the files in folders **test, file, pro_index, go_index**. Otherwise the program will reason based on these pre-existing files.

Rewrite:

- *_pro2go.npy - proteinid -> GO list dictionary in the training data
- pmid2text.npy - Pubmed ID -> title and abstract dictionary for the PubMed article you will use in the training and testing data
- proid2name.npy - proteinid -> protein name dictionary for the PubMed article you will use in the training and testing data
- go_index, pro_index - index built by pyserini

Delete (will be rebuilt after running predict.py):

- *_dev_t5_texts.npy - extracted sentences
- *_retrieval_all.npy
- *_retrieval_pro_3.npy - retrieval result
- *_t5_scores.npy - score cache

# File Structure

├── cross_model - Download

└────── *_PubMedBERT_epoch{}

└────── *_PubMedBERT_epoch1

├── data - training and testing data

└────── golden - evaluate files

└────── *_train.txt - train set

└────── test.txt - test set

├── file

└────── *_pro2go.npy - proteinid -> GO list

└────── pmid2text.npy - Pubmed ID -> title and abstract 

└────── proid2name.npy - proteinid -> protein name

├── test

└────── *_dev_t5_texts.npy - extracted sentences

└────── *_retrieval_all.npy

└────── *_retrieval_pro_3.npy - retrieval result

└────── *_t5_scores.npy - score cache

├── go_index

├── pro_index

├── result - final result

├── predict.py

└── data_preprocess.py
