# Skeleton-to-Response

### Requirement 

pytorch==0.3.1

### Usage

- Cascaded Model
  1. pretrain skeleton generator: go to the `template` folder, use `train.sh`
  2. pretrain response generator: go to the `pretrain` folder, use `train.sh`.
  3. train both with RL: go to the `hard` folder, use `train.sh`
  4. Test: go for `hard/translate.sh`
- Joint Model
  - Use the `train.sh` and `translate.sh` in the `soft` folder

### Data

The data we used in our paper are from [Wu et al, 2019](https://github.com/MarkWuNLP/ResponseEdit)

some sample data are in the `data` folder. The format is 

`query | response | retrieved query | retrieved response`

(sentences in each line are split by the symbol `|`)

### Citation

```
@inproceedings{cai-etal-2019-skeleton,
    title = "Skeleton-to-Response: Dialogue Generation Guided by Retrieval Memory",
    author = "Cai, Deng  and Wang, Yan  and Bi, Wei  and Tu, Zhaopeng  and Liu, Xiaojiang  and Lam, Wai  and Shi, Shuming",
    booktitle = "Proceedings of the 2019 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)",
    month = jun,
    year = "2019",
    address = "Minneapolis, Minnesota",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/N19-1124",
    pages = "1219--1228"
}
```
