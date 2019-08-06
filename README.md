# MTFN-RR PyTorch Code
The offical PyTorch code for paper "Matching Images and Text with Multi-modal Tensor Fusion and Re-ranking", ACM Multimedia 2019


## Introduction
This is the MTFN-RR, the PyTorch source code for the paper "Matching Images and Text with Multi-modal Tensor Fusion and Re-ranking" from UESTC and the author WangTan is a third-year undergraduated student. The paper will appear in Multimedia 2019, Nice, France. We thanks [MUTAN](https://github.com/cadene/vqa.pytorch) code for great help.

### What task does our code (method) solve?
Our MTFN-RR focuses on the image-text matching problem, that is, retrieving the relevant instances in a different media type from the query, including two typical cross-modal retrieval scenarios: 1) sentence retrieval (I2T), i.e., retrieving groundtruth sentences given a query image; 2) image retrieval (T2I), i.e., retrieving matched images given a query text.

### Insight of our model:
- Existing image-text matching approaches for pursuing this challenge generally fall into two categories: the embedding-based methods and the classification-based methods. However both approaches have notable problems (More details can be referred to our paper) and we are the first to try to fusion these two approaches and thier advantages.
- We propose a cross-modal reranking module inspired by the reranking scheme in unimodal task. It's a lightweight, easy-used, unsupervised arithmetic module that can be embedded into almost all image-text matching methods, to improve the performance in **a few seconds**.
- Our model has achieved the best performance on MSCOCO and Flickr30k dataset on cross-modal retrieval task.




## Installation and Requirements
### Installation
We recommended the following dependencies:
- Python 3
- PyTorch > 0.3
- Numpy
- h5py
- nltk
- yaml

### Other Modules
Our code also conclude some other modules for directing use
- [Skip-thoughts.torch](https://github.com/Cadene/skip-thoughts.torch)



## The Cross-modal Reranking (RR) moduel
Here we provide the original cross-modal reranking script `rerank.py` for directly using or modification.
- The code contains two parts:
  - The re-ranking arithmetic
  - The recall accuracy computing code

- The Input (The 1-stage prediction image-text similarity matrix)
   - You need prepare the image-text similarity matrix by the training model (any other methods)
   
     *The similarity is between 0 and 1, the address need to be revised in the py document*
   - The matrix is excepted to have the size of d x 5d (image x sentence)
   
     *Here we assume that 5 sentences associated to 1 iamge*
     
- The Output
  - The Recall@1, 5, 10 score before and after cross-modal reranking
  
- Usage
  ```
  python rerank.py
  ```
  
## Training
### Data Download
Download the dataset files and pre-trained models. We use splits produced by [Andrej Karpathy](http://cs.stanford.edu/people/karpathy/deepimagesent/). The raw images can be downloaded from from their original sources [here](http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/KCCA.html), [here](http://shannon.cs.illinois.edu/DenotationGraph/) and [here](http://mscoco.org/).

The precomputed image features of MS-COCO are from [here](https://github.com/peteanderson80/bottom-up-attention). The precomputed image features of Flickr30K are extracted from the raw Flickr30K images using the bottom-up attention model from [here](https://github.com/peteanderson80/bottom-up-attention). All the data needed for reproducing the experiments in the paper, including image features and vocabularies, can be downloaded from:

```
wget https://scanproject.blob.core.windows.net/scan-data/data.zip
wget https://scanproject.blob.core.windows.net/scan-data/vocab.zip
```

We refer to the path of extracted files for `data.zip` as `$DATA_PATH` and files for `vocab.zip` to `./vocab` directory. Alternatively, you can also run vocab.py to produce vocabulary files. For example,

```
python vocab.py --data_path data --data_name coco_precomp
```

Or we also provide the flickr and coco vocabulary json document in `vocab` folder and you just need to modify the address in `train.py`

### Running
After modify the hype parameters for dataset address, model save address and so on, you can just run following command. The hype parameters about model is in `option/FusionNoattn_baseline.yaml`
```
python train.py
```

### Code Structure
```
├── MTFN-RR/
|   ├── engine.py           /* Files contain train/validation code
|   ├── model.py            /* Files for the model layer
|   ├── data.py             /* Files for construct dataset
|   ├── utils.py            /* Files for tools
|   ├── train.py             /* Files for main code
|   ├── re_rank.py         /* Files for re-ranking in testing stage
|   ├── vocab.py           /* Files for construct vocabulary
|   ├── seq2vec.py         /* Files for sentence to vector
|   ├── readme.md
│   ├── option/               /* setting file
|   ├── vocab/               /vocabuary documents
```
