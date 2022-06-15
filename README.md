# Evaluating Attribution Methods for Explainable NLP with Transformers [TSD 2022]

## Datasets

Our modified Stanford Sentiment Treebank is in <code>datasets_ours/sst</code>.
There are three splits - <code>train.csv</code>, <code>test.tsv</code> and <code>dev.csv</code>. The preprocessed original files split into multiple <code>json</code> files.

The PMI values for Czech Text Document Corpus are in <code>datasets_ours/news/PMI.csv</code> and are a prt of this repository.
The CTDC dataset can be downloaded from [here](http://ctdc.kiv.zcu.cz/).
Then copy the <code>czech_text_document_corpus_v20</code> folder in the archive to <code>datasets_ours/news</code>.

## Training
### Stanford Sentiment Treebank
For SST training we use the HuggingFace [GLUE training script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification).
```bash
python run_glue.py \
  --model_name_or_path bert-base-cased \
  --do_train \
  --do_eval \
  --max_seq_length 512 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --learning_rate 1e-5 \
  --num_train_epochs 4 \
  --output_dir bert-base-cased-sst \
  --logging_strategy epoch \
  --validation_file datasets_ours/sst/dev.csv \
  --train_file datasets_ours/sst/train.csv \
  --save_strategy epoch \
  --evaluation_strategy epoch
```
### Czech Text Document Corpus
There are two stages in CTDC training. In line with the original authors, we first perform kfold cross-validation and then train our models on the entire dataset.

The kfold cross-validation:
```bash
python train_news_kfold.py \
  --model_name UWB-AIR/Czert-B-base-cased \
  --epochs 4 \
  --lr 1e-5 \
  --batch_size 4 \
  --output_dir output_kfold \
  --from_tf True
```

The training follows the same syntax:
```bash
python train_news.py \
  --model_name UWB-AIR/Czert-B-base-cased \
  --epochs 4 \
  --lr 1e-5 \
  --batch_size 4 \
  --output_dir Czert-B-base-cased-news \
  --from_tf True
```

## Attributions

Later on, when the attributions are evaluated, the evaluation script relies on <b>cased</b> or <b>uncased</b> being present in the directory name
in order to correctly merge the tokenized text into words. The medium, small and mini models are all uncased. Keep this in mind, as otherwise the script will crash. 

### SST

#### Baselines
For the SST dataset attributions we need neutral baselines.
Since this is a binary classification task, we can use gradients w.r.t. embeddings to generate neutral samples.
This can take a while, so in case the gradients are too slow, random generation can be used. Scripts for generating baselines
are in the <code>baselines</code> directory.

The usage is as follows:
```bash
python generate_neutral_baselines_sst.py \
  --tokenizer bert-base-cased \
  --model_file bert-base-cased-sst/pytorch_model.bin \  # for example
  --output_dir baselines_bert-base-cased-sst
```

The syntax is the same for random generation.

#### Attributions generation
The script <code>create_attributions_sst.py</code> generates attributions for the SST dataset.
The usage is as follows:
```bash
python create_attributions_sst.py \
  --baselines_directory baselines_czert \
  --model_path bert-base-cased-sst \  # for example
  --output_dir bert-base-cased-sst-attrs 
```

Be aware that multiple large files (typically in excess of 1GB) are generated.

### CTDC

For CTDC we use a dummy class which is injected into training and serves as a baseline which is neutral to all other classes.
No baselines need to be generated.

The script <code>create_attributions_news.py</code> generates attributions for the CTDC dataset.
Because this process can take a significant amount of time, there is an option to split the computation into different parts that can then run in parallel. See the script help for options.

The usage is as follows:
```bash
python create_attributions_news.py \
  --model_path Czert-B-base-cased-news \  # for example
  --output_dir Czert-B-base-cased-news-attrs \
  --part ig200  # optional, for no split use 'all'
```

Be aware that the files generated have sizes in excess of 10GB.

## Evaluation

### SST

For SST evaluation use the <code>process_attributions_sst.py</code> script:
```bash
python process_attributions_sst.py \
  --attrs_dir bert-base-cased-sst-attrs \  # for example
  --output_file metrics_sst.csv \
  --absolute True  # True or False, by default False
```

### CTDC

For CTDC evaluation use the <code>process_attributions_news.py</code> script:
```bash
python process_attributions_sst.py \
  --attrs_dir Czert-B-base-cased-news-attrs \  # for example
  --output_file metrics_news.csv \
  --absolute True  # True or False, by default False
```

## Our models

### SST

For SST, we have used the following pretrained models from HuggingFace:
* [bert-base-cased](https://huggingface.co/bert-base-cased)
* [prajjwal1/bert-medium](https://huggingface.co/prajjwal1/bert-medium)
* [prajjwal1/bert-small](https://huggingface.co/prajjwal1/bert-small)
* [prajjwal1/bert-mini](https://huggingface.co/prajjwal1/bert-mini)

Our trained versions are:
* [bert-base-cased](https://drive.google.com/file/d/10LiYqr8HL3Zhpy-k4izKlLh57ZjPz-w9/view?usp=sharing)
* [prajjwal1/bert-medium](https://drive.google.com/file/d/1YPwFFUn_Grm6zq18GPmnlRH9dZnfS1_s/view?usp=sharing)
* [prajjwal1/bert-small](https://drive.google.com/file/d/1v70e0ScMMfIZWyti4aS_ZTvNZEYk8s2M/view?usp=sharing)
* [prajjwal1/bert-mini](https://drive.google.com/file/d/1tvUx31QC6WAjhdCjy_ZMZ6h_iEnI1REz/view?usp=sharing)

### CTDC

For CTDC, we have used the following models from HuggingFace:
* [UWB-AIR/Czert-B-base-cased](https://huggingface.co/UWB-AIR/Czert-B-base-cased)
* [Seznam/small-e-czech](https://huggingface.co/Seznam/small-e-czech)

Our trained versions are:
* [UWB-AIR/Czert-B-base-cased](https://drive.google.com/file/d/19nGVbkb46XqqMy4Z7f3C881fTw8OgEKq/view?usp=sharing)
* [Seznam/small-e-czech](https://drive.google.com/file/d/1wdubynicCkcAXr_zZUEODIM6dZOm0OE-/view?usp=sharing)


# Notes

We have used the reference implementation from Chefer et al. [Transformer Interpretability Beyond Attention Visualization](https://github.com/hila-chefer/Transformer-Explainability) in order to evaluate their method.