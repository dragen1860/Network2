# LearningToCompare
Pytorch Implementation for Paper: Learning to Compare: Relation Network for Few-Shot Learning

# Howto
```python
python train.py
```

# NOTICE
current code support multi-gpus on single machine training, to disable it and train on single machine, 
just set device_ids=[0] and downsize batch size according to your gpu memory capacity.
make sure `ckpt` directory exists, otherwise `mkdir ckpt`.


# mini-Imagenet

## Deeper version

| Model                               | Fine Tune | 5-way Acc. |        | 20-way Acc |        |
|-------------------------------------|-----------|------------|--------|------------|--------|
|                                     |           | 1-shot     | 5-shot | 1-shot     | 5-shot |
| Matching Nets                       | N         | 43.56%     | 55.31% | 17.31%     | 22.69% |
| Meta-LSTM                           |           | 43.44%     | 60.60% | 16.70%     | 26.06% |
| MAML                                | Y         | 48.7%      | 63.11% | 16.49%     | 19.29% |
| Meta-SGD                            |           | 50.49%     | 64.03% | 17.56%     | 28.92% |
| TCML                                |           | 55.71%     | 68.88% | -          | -      |
| Learning to Compare           	  | N         | 57.02%     | 71.07% | -          | -      |
| ***reproduction*				      | N         |  55.2%     |    68.8% |          |        | 
| ***Ours*				     		  | N         |  53.0%     |    64.6% |          |        | 

## Naive version

| Model                               | Fine Tune | 5-way Acc. |        | 20-way Acc |        |
|-------------------------------------|-----------|------------|--------|------------|--------|
|                                     |           | 1-shot     | 5-shot | 1-shot     | 5-shot |
| Matching Nets                       | N         | 43.56%     | 55.31% | 17.31%     | 22.69% |
| Meta-LSTM                           |           | 43.44%     | 60.60% | 16.70%     | 26.06% |
| MAML                                | Y         | 48.7%      | 63.11% | 16.49%     | 19.29% |
| Meta-SGD                            |           | 50.49%     | 64.03% | 17.56%     | 28.92% |
| Learing to compare                          |     N      | 51.38%     |67.07%| -    | - |
| naivern.py      (naive version)                    |     N      | 53.8%     |	67.5%	| -    | - |


* 6x6 spational relation seems good, meaning larger spatial will lead to better
* naive version overcome all, meaning simplier nn 
* the conv after repnet will occupy very huge gpu memory, but only reduce it will not lead to good, 51% on 5-way-1shot
* 