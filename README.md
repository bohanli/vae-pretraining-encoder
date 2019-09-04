# A Surprisingly Effective Fix for Deep Latent Variable Modeling of Text

This is PyTorch implementation of the following [paper](https://arxiv.org/abs/1909.00868):

```
A Surprisingly Effective Fix for Deep Latent Variable Modeling of Text
Bohan Li*, Junxian He*, Graham Neubig, Taylor Berg-Kirkpatrick, Yiming Yang
EMNLP 2019
```

Please contact bohanl1@cs.cmu.edu if you have any questions.

## Requirements

* Python >= 3.6
* PyTorch >= 1.0
* pip install editdistance

## Data

Datasets used in this paper can be downloaded with:

```
python prepare_data.py
```

## Usage

Train a AE first
```
python text_beta.py \
    --dataset yahoo \
    --beta 0 \
    --lr 0.5
```

Train VAE with our method
```
ae_exp_dir=exp_yahoo_beta/yahoos_lr0.5_beta0.0_drop0.5
python text_anneal_fb.py \
    --dataset yahoo \
    --load_path ${ae_exp_dir}/model.pt \
    --reset_dec \
    --kl_start 0 \
    --warm_up 10 \
    --target_kl 8 \
    --fb 2 \
    --lr 0.5
```

Logs, models and samples would be saved into folder `exp`.


## Reference

```
@inproceedings{li2019emnlp,
    title = {A Surprisingly Effective Fix for Deep Latent Variable Modeling of Text},
    author = {Bohan Li and Junxian He and Graham Neubig and Taylor Berg-Kirkpatrick and Yiming Yang},
    booktitle = {Conference on Empirical Methods in Natural Language Processing (EMNLP)},
    address = {Hong Kong},
    month = {November},
    year = {2019}
}

```

## Acknowledgements

A large portion of this repo is borrowed from https://github.com/jxhe/vae-lagging-encoder

