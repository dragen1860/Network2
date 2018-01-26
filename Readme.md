
# Ominiglot
| Model                               	| Fine Tune 	| 5-way Acc.    	|               	| 20-way Acc   	|               	|
|-------------------------------------	|-----------	|---------------	|---------------	|--------------	|---------------	|
|                                     	|           	| 1-shot        	| 5-shot        	| 1-shot       	| 5-shot        	|
| MANN                                	| N         	| 82.8%         	| 94.9%         	| -            	| -             	|
| Matching Nets                       	| N         	| 98.1%         	| 98.9%         	| 93.8%        	| 98.5%         	|
| Matching Nets                       	| Y         	| 97.9%         	| 98.7%         	| 93.5%        	| 98.7%         	|
| MAML                                	| Y         	| 98.7+-0.4%    	| 99.9+-0.1%    	| 95.8+-0.3%   	| 98.9+-0.2%    	|
| Meta-SGD                            	|           	| 99.53+-0.26%  	| 99.93+-0.09%  	| 95.93+-0.38% 	| 98.97+-0.19%  	|
| TCML                                	|           	| 98.96+-0.20% 	| 99.75+-0.11% 	| 97.64+-0.30% 	| 99.36+-0.18% 	|
| Learning to Compare                 	| N         	| 99.6+-0.2%   	| 99.8+-0.1%    	| 97.6+-0.2%   	| 99.1+-0.1%    	|
| Ours(Res18, flatten features, 2 fc) 	| Y         	| 98.99% ,48120ep, 20b            	|       99.6%,48620ep, 5b         	|    96.99%,153920ep,20b            	|   97.2%, 63220ep,2b         	|
|naive, omni.py  | N | 99.6% | 99.8% |  98.88% |   |





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
| naivern.py      (naive version, avg pool, 9e-4)                    |     N      | 56.0%->60.8%, 2days     |	68.1%	| -    | - |


## Simplified Deep version

| Model                               | Fine Tune | 5-way Acc. |        | 20-way Acc |        |
|-------------------------------------|-----------|------------|--------|------------|--------|
|                                     |           | 1-shot     | 5-shot | 1-shot     | 5-shot |
| Matching Nets                       | N         | 43.56%     | 55.31% | 17.31%     | 22.69% |
| Meta-LSTM                           |           | 43.44%     | 60.60% | 16.70%     | 26.06% |
| MAML                                | Y         | 48.7%      | 63.11% | 16.49%     | 19.29% |
| Meta-SGD                            |           | 50.49%     | 64.03% | 17.56%     | 28.92% |
| TCML                                |           | 55.71%     | 68.88% | -          | -      |
| Learning to Compare           	  | N         | 57.02%     | 71.07% | -          | -      |
| rn.py, 463bottleneck, 6x6, conv->maxpool                    |     N      | 53.3%     |		| -    | - |
| simrn.py, 111basicneck                   |     N      | 53.0%     |		| -    | - |
 


* 6x6 spational relation seems good, meaning larger spatial will lead to better
* naive version overcome all, meaning simplier nn 
* the conv after repnet will occupy very huge gpu memory, but only reduce it will not lead to good, 51% on 5-way-1shot
* avg pooling
* sum on feature can not converge
* g network can enlarge since it have more batch
