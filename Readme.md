
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
|naive, omni.py, 2days  | N | 99.80% | 100% |  98.88% |   |




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
| naivern.py      (naive version)     |     N      | 53.8%     |	67.5%	| -    | - |
| naivern.py      (naive version, avg pool, 9e-4) |     N      | 56.0%->60.8%, 2days     |	68.1%	| -    | - |
| naive5.py      (naive version, avg pool, sum over features) |     N      |      |	72.7| -    | - |
| naivesum.py      (naive version, avg pool, sum over features, concat all setsz after f) |     N      |      |	70.8| -    | - |


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
| rn.py, 463bottleneck, 6x6, conv->maxpool   |     N      | 53.3%     |		| -    | - |
| simrn.py, 111basicneck                   |     N      | 55.4%     |	66.7%	| -    | - |
 


* 6x6 spational relation seems good, meaning larger spatial will lead to better
* naive version overcome all, meaning simplier nn 
* the conv after repnet will occupy very huge gpu memory, but only reduce it will not lead to good, 51% on 5-way-1shot
* avg pooling
* sum on feature can not converge
* g network can enlarge since it have more batch
* spatial d is very critical, small d lead to faster converge, however, when it read 50%, it read the roof. Large `d` will converge slow but 
have higher possibilityes.
* reduce Naive 4 conv to 3 conv will not improve performance(55.7%), and reduce d of spatial rn will not improve as well().
* add BatchNorm1d in self.g is extremely slow, will cost 3x time compared with none BN.
* add 4conv to 6conv will not improve: 68.7%
* smaller rn input, as 5x5, bad
* 12x12, no last layer avgpool, randomhorizontal flip, 65.9%
* deep version of relation, 64channel all, out 5x5 67.5%



>naivesum 
0.712, 0.717, 0.736, 0.643, 0.672, 0.661, 0.653, 0.739, 0.677, 0.736, 0.741, 0.701, 0.603, 0.661, 0.693, 0.749, 0.709, 0.739, 0.659, 0.720, 0.688, 0.627, 0.717, 0.624, 0.715, 0.712, 0.741, 0.707, 0.704, 0.683, 0.683, 0.701, 0.728, 0.707, 0.715, 0.685, 0.720, 0.789, 0.680, 0.651, 0.664, 0.720, 0.784, 0.669, 0.744, 0.712, 0.600, 0.755, 0.717, 0.648, 0.595, 0.715, 0.731, 0.723, 0.680, 0.677, 0.699, 0.755, 0.688, 0.803, 0.707, 0.672, 0.651, 0.717, 0.747, 0.707, 0.744, 0.696, 0.712, 0.669, 0.717, 0.651, 0.704, 0.672, 0.696, 0.699, 0.645, 0.691, 0.768, 0.648, 0.685, 0.653, 0.616, 0.675, 0.629, 0.685, 0.736, 0.651, 0.691, 0.739, 0.635, 0.632, 0.707, 0.685, 0.693, 0.661, 0.643, 0.685, 0.699, 0.664, 0.773, 0.677, 0.709, 0.712, 0.659, 0.659, 0.733, 0.739, 0.787, 0.643, 0.747, 0.669, 0.643, 0.699, 0.733, 0.680, 0.680, 0.664, 0.704, 0.683, 
accuracy: 0.694533333333 sem: 0.00744224200062

>slave2: original version, train for 24hours
0.707, 0.754, 0.689, 0.677, 0.677, 0.701, 0.704, 0.733, 0.738, 0.674, 0.677, 0.723, 0.717, 0.724, 0.708, 0.686, 0.702, 0.684, 0.674, 0.696, 0.744, 0.727, 0.793, 
accuracy: 0.709146537842 sem: 0.0130249463636


>gpu: g+fc, f+fc
0.723, 0.750, 0.670, 0.693, 0.703, 0.707, 0.737, 0.727, 0.710, 0.673, 0.750, 0.673, 0.780, 0.650, 0.703, 0.623, 0.773, 0.823, 0.557, 0.693, 0.717, 0.720, 0.707, 0.667, 0.753, 
accuracy: 0.707333333333 sem: 0.0222004602888

>p100:
0.627, 0.690, 0.637, 0.703, 0.700, 0.700, 0.667, 0.617, 0.690, 0.673, 0.723, 0.680, 0.720, 0.750, 0.697, 0.740, 0.713, 0.777, 0.693, 0.670, 0.743, 0.770, 0.730, 0.730, 0.713, 
accuracy: 0.702133333333 sem: 0.0167383742684

>5way1shot fine tune from 60.8
0.553, 0.560, 0.687, 0.583, 0.560, 0.570, 0.610, 0.563, 0.523, 0.543, 0.560, 0.590, 0.603, 0.570, 0.620, 0.647, 0.610, 0.517, 0.493, 0.597, 0.507, 0.623, 0.640, 0.537, 0.617, 
accuracy: 0.579333333333 sem: 0.01934070244

0.597, 0.720, 0.597, 0.557, 0.543, 0.473, 0.680, 0.640, 0.620, 0.640, 0.503, 0.533, 0.593, 0.507, 0.567, 0.583, 0.677, 0.637, 0.617, 0.507, 0.643, 0.647, 0.630, 0.497, 0.550, 
accuracy: 0.590266666667 sem: 0.0268392648054

>metric
0.69067, 0.72000, 0.68533, 0.68800, 0.70133, 0.70933, 0.70933, 0.79200, 0.72267, 0.74400, 0.68533, 0.72533, 0.58667, 0.73867, 0.73067, 0.71200, 0.70400, 0.71200, 0.68800, 0.70667, 0.70933, 0.76533, 0.74400, 0.72267, 0.68267, 0.69333, 0.73067, 0.73067, 0.66133, 0.68800, 0.69600, 0.74133, 0.69600, 0.73333, 0.70667, 0.72000, 0.74400, 0.67200, 0.76533, 0.65333, 0.66133, 0.76533, 0.69067, 0.72533, 0.62933, 0.66133, 0.66667, 0.72267, 0.70933, 0.68800, 0.71467, 0.68533, 0.72000, 0.72000, 0.71733, 0.71200, 0.65600, 0.74400, 0.74933, 0.67200, 0.71733, 0.69067, 0.68800, 0.70667, 0.76800, 0.67733, 0.68267, 0.75733, 0.69867, 0.68800, 0.74667, 0.67733, 0.70933, 0.67200, 0.67200, 0.68800, 0.66933, 0.64267, 0.67200, 0.74667, 0.77333, 0.72533, 0.77067, 0.72000, 0.74133, 0.71200, 0.78400, 0.68533, 0.67200, 0.67467, 0.68533, 0.75467, 0.69867, 0.64533, 0.66400, 0.73067, 0.77067, 0.69333, 0.62667, 0.68000, 0.71467, 0.63733, 0.63733, 0.70133, 0.67200, 0.64267, 0.66933, 0.73600, 0.62933, 0.69600, 0.67200, 0.71200, 0.68000, 0.62933, 0.70933, 0.70933, 0.73867, 0.73067, 0.67733, 0.73333, 
accuracy: 0.702711111111 sem: 0.00676532628843
<<<<<<<<< accuracy: 0.702711111111 best accuracy: 0.699428571429 >>>>>>>>


>learnign2compare: 5way5shot:
0.63111, 0.66222, 0.73333, 0.72444, 0.68889, 0.70222, 0.71111, 0.70222, 0.71556, 0.66222, 0.76889, 0.64000, 0.71111, 0.67111, 0.60889, 0.68889, 0.68444, 
accuracy: 0.6886274509803921 sem: 0.02055769936837831

>slave2 5way 1shot:0.54133, 0.64533, 0.52533, 0.55733, 0.58400, 0.58133, 0.61067, 0.55467, 0.62400, 0.61867, 0.55733, 0.59200, 0.65333, 0.62667, 0.50667, 0.59200, 0.57067, 0.62133, 0.53867, 0.64533, 0.56800, 0.54400, 0.55733, 
accuracy: 0.583304347826 sem: 0.0178551543546

>20way 1shot: 0.98333, 0.99667, 0.99333, 0.96667, 0.99667, 0.97667, 0.98000, 1.00000, 0.98333, 0.98667, 0.99667, 0.96667, 0.98000, 0.99667, 0.96333, 0.98000, 0.99667, 0.99667, 0.98667, 0.99333, 0.99333, 1.00000, 0.99000, 0.99000, 0.99000, 1.00000, 0.98667, 0.99667, 0.99333, 0.99333, 0.99667, 0.96000, 0.96667, 0.98667, 0.99333, 0.97333, 0.99667, 0.97333, 0.99333, 0.98667, 0.97667, 0.99667, 0.99667, 0.98333, 0.97667, 0.99333, 0.98333, 0.97000, 0.98667, 0.95667, 0.98000, 0.97333, 0.99667, 0.97333, 0.98667, 0.98000, 0.99667, 0.96000, 0.97000, 0.96667, 0.99333, 0.96667, 0.99000, 0.98667, 0.99000, 0.99000, 0.99333, 1.00000, 0.99000, 0.98333, 0.98000, 0.96333, 0.99000, 0.99333, 0.98667, 0.97000, 0.97333, 0.98333, 0.99000, 0.97000, 0.99667, 0.99667, 0.99333, 0.97667, 0.98000, 0.99667, 0.96667, 0.97667, 0.98000, 0.98000, 0.99000, 0.99667, 0.97000, 0.99000, 0.99333, 0.99333, 0.98667, 0.99333, 0.99333, 0.99333, 0.98667, 0.97333, 0.97667, 0.99333, 0.98667, 0.97667, 0.99000, 0.91000, 0.98667, 0.98000, 0.93333, 0.97667, 0.98333, 0.99333, 1.00000, 0.98333, 0.99667, 0.98000, 1.00000, 0.99333, 0.97667, 0.98000, 0.99333, 0.98667, 0.99000, 0.99333, 1.00000, 0.99667, 0.99000, 0.98333, 1.00000, 0.98000, 0.98667, 0.99333, 0.93667, 0.98333, 0.98000, 0.99000, 0.98667, 0.97667, 0.95000, 0.98667, 0.98667, 0.98000, 1.00000, 0.98000, 0.94667, 0.98667, 0.97333, 0.99333, 0.99667, 0.99000, 0.98667, 0.98000, 0.99667, 0.99000, 0.97667, 0.99333, 0.97667, 1.00000, 0.97000, 0.98333, 0.97333, 0.99000, 0.95333, 0.97667, 0.94333, 0.99000, 0.99000, 0.99333, 0.98000, 0.99667, 0.95000, 0.99000, 0.98333, 0.96667, 0.99333, 0.96333, 0.99000, 0.98333, 0.98333, 0.99000, 1.00000, 0.99000, 0.99667, 0.99000, 0.98667, 0.99333, 0.97667, 0.96000, 0.97667, 1.00000, 0.99333, 0.99000, 0.98667, 0.98667, 0.97333, 0.98333, 0.99333, 0.99333, 0.99000, 0.98333, 0.93667, 0.99333, 0.99000, 0.98667, 0.97000, 0.98667, 0.97667, 0.99333, 0.99667, 0.99667, 0.99000, 0.99000, 0.97667, 0.99333, 0.95333, 0.98333, 0.99000, 0.99333, 0.94000, 0.99000, 0.99333, 0.98667, 0.98333, 1.00000, 0.99667, 0.99000, 0.98667, 0.97333, 0.97333, 0.96000, 0.99667, 0.99333, 0.97333, 0.99333, 1.00000, 0.99000, 0.97000, 0.98667, 0.99667, 0.93333, 0.99667, 0.98333, 0.98000, 0.96333, 0.95333, 0.99333, 0.96000, 0.97000, 0.99333, 0.95667, 1.00000, 0.98333, 0.99667, 0.99333, 0.97667, 0.98667, 0.99333, 0.96000, 0.98000, 1.00000, 0.97000, 0.99000, 0.94667, 0.92333, 0.98333, 0.99667, 0.97000, 0.99667, 1.00000, 0.99333, 0.97667, 0.98000, 0.99000, 0.98333, 0.99333, 0.99667, 0.99667, 0.99667, 0.96333, 0.99667, c0.97333, 0.96333, 0.93667, 0.98000, 0.98000, 0.99000, 0.98333, 0.99667, 0.92000, 0.98667, 0.94667, 0.98667, 0.97000, 0.97333, 0.98667, 0.97000, 0.99667, 0.99333, 0.99000, 0.99667, 1.00000, 0.98000, 0.99000, 0.99667, 0.99667, 0.96333, 0.89667, 0.93333, 0.99667, 1.00000, 0.98667, 0.99333, 0.98000, 0.97000, 0.99667, 0.99333, 0.99000, 0.98667, 0.99333, 0.99667, 0.99667, 0.98000, 0.99667, 0.99000, 0.97000, 0.99000, 0.98667, 0.99667, 0.99333, 0.99000, 0.99667, 0.98333, 0.98000, 0.97000, 0.96333, 0.97000, 0.98333, 0.99000, 0.99333, 0.99667, 0.98667, 1.00000, 0.97333, 0.98667, 0.98333, 0.99000, 0.99333, 0.98000, 0.98333, 0.97667, 0.99667, 0.93333, 0.99333, 0.95667, 0.99333, 0.97667, 0.99333, 0.99333, 0.93333, 0.98667, 0.98667, 0.96000, 0.99667, 0.97333, 0.99000, 0.97333, 0.91667, 0.99000, 0.98667, 0.96333, 0.95333, 1.00000, 0.93667, 0.99000, 1.00000, 0.95000, 0.99333, 0.98000, 0.99333, 0.94333, 0.98333, 0.99000, 0.97667, 0.98667, 0.97667, 0.98667, 0.99333, 0.98000, 0.99333, 0.96667, 0.99333, 0.96000, 1.00000, 0.97667, 0.96000, 0.96333, 0.99333, 1.00000, 0.97000, 0.99000, 0.98333, 0.99333, 0.98333, 0.97000, 0.97333, 0.99000, 0.96667, 0.99333, 0.98667, 0.97667, 0.99333, 0.98000, 0.99000, 0.99333, 0.99333, 0.97667, 0.93000, 0.95000, 0.97333, 1.00000, 1.00000, 0.99667, 1.00000, 0.97667, 0.99333, 0.97333, 0.99000, 1.00000, 0.98333, 1.00000, 0.99333, 0.99333, 0.96667, 1.00000, 0.97667, 0.94667, 0.98333, 0.98333, 0.98000, 0.97333, 0.99000, 0.99667, 0.99000, 0.99667, 0.97333, 0.95667, 0.98333, 0.98333, 0.98333, 0.99000, 0.99000, 0.97333, 0.99000, 0.98667, 0.93333, 0.99667, 0.97000, 0.99333, 0.98333, 0.98667, 0.99333, 0.98333, 0.99333, 0.98000, 0.93667, 0.98333, 0.98333, 0.99333, 0.96333, 0.98000, 0.99000, 0.94333, 0.98333, 0.99000, 0.98667, 0.99667, 0.97000, 0.97000, 0.98667, 0.97667, 0.99000, 0.94667, 0.98000, 0.98000, 0.97000, 0.99000, 0.99667, 0.97667, 0.98333, 0.99333, 0.97333, 0.99333, 0.99000, 0.95667, 0.99667, 0.93333, 0.98667, 0.97667, 0.94333, 0.99333, 0.99667, 0.99667, 0.95000, 0.98000, 0.99000, 0.96333, 0.94333, 0.98667, 0.97333, 0.99000, 0.97667, 1.00000, 0.99000, 0.97667, 1.00000, 0.99667, 0.97333, 0.95667, 0.98667, 0.98333, 0.98333, 0.99333, 0.97667, 0.99667, 0.97667, 0.96667, 0.99000, 0.99333, 0.99000, 0.99000, 0.99667, 0.97667, 0.98000, 0.98667, 0.98667, 0.95000, 0.98333, 0.95667, 0.97333, 0.98667, 1.00000, 0.99000, 0.96333, 0.99667, 0.92000, 0.99667, 0.99000, 0.99333, 0.94000, 0.99000, 1.00000, 0.97333, 0.98667, 0.99000, 0.99333, 0.97000, 1.00000, 0.99333, 0.99333, 0.98333, 0.98333, 0.99000, 0.96667, 0.99667, 0.99333, 0.99667, 0.95333, 0.99333, 0.94333, 0.99333, 0.99000, 0.99000, 1.00000, 0.96000, 0.96000, 0.99667, 0.99667, 0.99333, 0.96333, 0.98000, 0.96000, 0.96333, 0.98667, 0.98667, 0.99333, 0.98667, 0.99333, 0.99000, 0.94333, 0.99667, 0.99333, 0.99333, 0.94000, 0.94667, 0.99667, 0.96333, 0.97000, 0.99667, 
accuracy: 0.981938888889 sem: 0.00131637273997

>20way5shot:
0.99667, 0.99667, 0.99889, 0.99889, 0.99667, 0.99111, 0.99778, 0.99556, 0.99778, 0.99778, 0.99667, 0.99778, 0.99444, 0.99222, 0.98556, 0.99000, 1.00000, 0.99889, 0.99444, 0.99556, 0.98667, 0.98556, 0.98222, 0.99556, 0.99778, 0.99778, 0.99444, 0.99889, 0.99111, 0.99111, 1.00000, 1.00000, 0.99778, 0.99667, 0.99667, 0.99222, 0.99556, 0.99444, 0.99333, 0.99444, 0.99778, 0.99778, 1.00000, 1.00000, 0.99444, 0.99222, 1.00000, 0.99778, 0.99556, 0.98556, 1.00000, 0.99556, 0.99778, 0.98444, 0.99222, 0.99333, 0.99444, 0.99667, 0.98889, 0.98333, 0.99000, 0.99889, 0.99778, 0.99778, 0.98667, 0.99444, 0.99333, 0.99444, 0.99778, 0.99444, 0.99111, 0.99556, 0.99889, 0.99222, 0.99667, 0.99444, 0.99667, 0.99333, 0.99667, 0.99778, 0.99667, 0.98889, 0.99556, 0.99222, 0.99111, 0.98111, 0.99222, 0.99556, 0.98111, 0.98111, 0.99444, 0.99667, 0.99111, 0.99667, 0.99667, 0.99778, 0.99556, 0.99667, 0.99444, 0.99111, 0.99667, 0.98667, 0.99444, 0.98222, 0.99667, 0.99889, 0.99444, 0.99667, 0.99889, 0.99333, 0.99778, 0.98778, 0.99556, 0.99444, 0.99667, 0.99667, 0.99444, 0.99222, 0.97556, 0.98667, 0.99667, 0.99556, 0.97667, 0.99556, 0.99556, 0.99333, 0.99778, 0.98889, 0.99333, 0.99556, 0.98667, 0.99667, 0.99667, 0.99667, 0.99333, 0.99889, 0.99889, 0.99667, 0.99333, 0.99778, 0.99333, 1.00000, 0.99444, 0.99889, 0.99778, 0.99444, 0.99222, 0.99556, 0.99667, 0.99667, 0.99889, 0.99778, 0.99333, 0.99556, 0.99778, 0.98111, 0.99667, 0.99444, 0.99667, 0.99667, 0.97556, 0.99556, 0.99778, 0.99222, 0.99222, 0.99889, 0.99778, 0.99444, 0.99667, 0.99778, 0.99556, 0.99444, 0.99444, 0.99889, 0.99222, 0.99667, 0.99556, 0.99444, 0.99556, 0.98889, 0.99667, 0.99222, 0.99556, 0.98111, 0.99556, 0.99889, 0.99667, 0.99333, 0.99667, 0.99444, 0.98667, 0.99222, 0.98778, 0.98778, 0.99444, 0.98889, 0.99889, 0.99778, 0.98778, 0.99444, 
accuracy: 0.994111111111 sem: 0.000670760856559

0.99400, 0.98667, 0.99867, 0.99333, 0.99600, 0.99333, 0.99733, 0.99933, 0.99400, 0.99533, 0.99600, 0.99800, 0.99333, 0.99733, 0.99400, 0.99533, 0.99533, 0.99533, 0.99533, 0.99800, 0.99733, 0.99600, 0.99733, 0.99333, 0.99667, 0.99533, 0.98867, 0.98933, 0.99600, 0.99000, 0.99400, 0.99467, 0.99067, 0.99533, 0.99400, 0.99600, 0.99200, 0.99533, 0.99733, 0.99333, 0.99800, 0.99667, 0.99400, 0.99733, 0.99267, 0.99533, 0.99667, 0.99133, 0.98400, 0.99200, 0.99333, 0.99533, 0.99467, 0.99800, 0.99200, 0.99533, 0.99867, 0.99467, 0.99733, 0.99600, 0.99933, 0.99267, 0.99600, 0.99667, 0.99733, 0.99600, 0.99467, 0.99533, 0.99800, 0.99667, 0.99667, 0.99733, 0.99533, 0.99867, 0.99733, 0.99867, 0.99267, 0.99067, 0.98933, 0.99667, 0.99533, 0.99000, 0.99133, 0.99133, 0.99067, 0.99467, 0.99800, 0.99267, 0.99800, 0.99867, 0.99267, 0.99533, 0.99533, 0.99600, 0.98600, 0.99200, 0.98733, 0.99467, 0.99333, 0.99067, 0.99800, 0.99800, 0.98400, 0.99400, 0.99333, 0.99800, 0.99533, 0.99733, 0.99533, 0.99733, 0.99867, 0.99600, 0.99800, 0.99667, 0.99333, 0.99200, 0.99667, 0.99733, 0.99467, 0.99600, 
accuracy: 0.994766666667 sem: 0.000560808026472



>5way5shot omniglot:
1.00000, 1.00000, 0.99333, 0.99667, 0.99333, 1.00000, 1.00000, 0.99667, 1.00000, 1.00000, 1.00000, 1.00000, 0.99667, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 0.99667, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 0.99667, 1.00000, 0.99667, 1.00000, 1.00000, 0.99333, 1.00000, 1.00000, 1.00000, 0.99667, 1.00000, 0.99667, 1.00000, 1.00000, 1.00000, 0.99667, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 0.99667, 1.00000, 0.99667, 1.00000, 0.99333, 1.00000, 0.99000, 0.99667, 0.99333, 1.00000, 1.00000, 0.99667, 1.00000, 1.00000, 0.99667, 0.99000, 1.00000, 1.00000, 1.00000, 1.00000, 0.99333, 0.99667, 0.99667, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 0.99667, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 0.99667, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 0.99333, 1.00000, 1.00000, 0.98667, 1.00000, 1.00000, 0.99667, 1.00000, 0.99667, 0.99667, 0.99333, 0.99667, 1.00000, 0.98667, 0.99667, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 0.99667, 0.99000, 1.00000, 1.00000, 1.00000, 1.00000, 0.99667, 0.99667, 0.99333, 1.00000, 1.00000, 0.99667, 1.00000, 0.99667, 1.00000, 1.00000, 1.00000, 0.99667, 0.99667, 1.00000, 0.99667, 1.00000, 1.00000, 0.98333, 1.00000, 1.00000, 0.99000, 0.99667, 1.00000, 0.99333, 0.99667, 1.00000, 1.00000, 1.00000, 1.00000, 
accuracy: 0.998266666667 sem: 0.000493455325904

>5way5shot omnitlogt:
0.99867, 1.00000, 0.99733, 0.99933, 0.99933, 0.99867, 0.99733, 0.99933, 0.99867, 0.99733, 0.99800, 0.99667, 1.00000, 0.99933, 0.99867, 0.99533, 0.99867, 0.99800, 0.99800, 0.99933, 0.99933, 0.99933, 0.99867, 0.99867, 0.99867, 0.99800, 0.99867, 0.99933, 0.99933, 0.99667, 
accuracy: 0.998488888889 sem: 0.000397297144601

>5way1shot omniglot:
0.99667, 1.00000, 0.99500, 0.99500, 0.99333, 1.00000, 1.00000, 1.00000, 1.00000, 0.99333, 0.99833, 0.99333, 0.99833, 0.99833, 0.99667, 0.99667, 1.00000, 0.99667, 0.99667, 1.00000, 1.00000, 0.98833, 1.00000, 0.99667, 0.99833, 0.99833, 0.99833, 0.99500, 0.99833, 0.99667, 0.99833, 0.99500, 1.00000, 1.00000, 0.99667, 1.00000, 0.99667, 1.00000, 0.99833, 1.00000, 1.00000, 1.00000, 0.99833, 0.99500, 0.99333, 0.99833, 0.97000, 0.99833, 0.99833, 0.99667, 1.00000, 0.99167, 0.99500, 0.99833, 0.99833, 0.99833, 0.98500, 0.99167, 1.00000, 0.99500, 1.00000, 0.99833, 0.99667, 0.99833, 0.99667, 0.99500, 0.99667, 1.00000, 0.99667, 0.99500, 0.99833, 0.99833, 0.99500, 0.99000, 0.99333, 
accuracy: 0.996711111111 sem: 0.000990718100947

1.00000, 0.99833, 0.99167, 1.00000, 0.99667, 0.99667, 1.00000, 0.98000, 0.99500, 1.00000, 0.99667, 0.99500, 0.99833, 0.99667, 1.00000, 1.00000, 0.98667, 0.99000, 0.99833, 0.99833, 1.00000, 0.99833, 0.99667, 1.00000, 1.00000, 0.99833, 1.00000, 1.00000, 1.00000, 0.99333, 0.99833, 0.99833, 0.99667, 0.98333, 1.00000, 0.99500, 0.99500, 1.00000, 0.99667, 0.99833, 0.99333, 0.99500, 0.99833, 1.00000, 0.99667, 0.99000, 0.99833, 0.99833, 0.99333, 0.99500, 0.99833, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 0.99833, 0.99667, 1.00000, 1.00000, 0.99167, 1.00000, 1.00000, 0.99833, 0.99667, 0.99333, 0.99500, 1.00000, 1.00000, 0.99833, 0.99833, 0.99833, 1.00000, 1.00000, 1.00000, 
accuracy: 0.997177777778 sem: 0.000891478983353

0.99833, 0.98833, 1.00000, 0.99667, 1.00000, 0.99667, 0.96833, 1.00000, 0.99667, 0.99500, 1.00000, 0.99833, 0.99833, 0.99833, 0.99667, 0.99833, 0.99833, 0.99500, 0.99833, 0.99167, 0.99833, 1.00000, 0.99667, 0.99833, 0.99500, 0.99667, 1.00000, 1.00000, 0.99833, 1.00000, 0.99667, 1.00000, 0.99833, 0.99833, 1.00000, 1.00000, 0.99833, 0.99833, 0.99833, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 0.99667, 1.00000, 1.00000, 0.99833, 0.99833, 1.00000, 0.99833, 0.99833, 0.99333, 0.99667, 1.00000, 1.00000, 0.99667, 0.99667, 0.99833, 0.99833, 0.99833, 0.99500, 1.00000, 1.00000, 0.99500, 0.99833, 1.00000, 0.99833, 1.00000, 0.99833, 0.99667, 0.99500, 0.99833, 0.99833, 0.99167, 
accuracy: 0.997577777778 sem: 0.00094202611958


>5wah1shot: imagenet:
0.52000, 0.66133, 0.57333, 0.59733, 0.55467, 0.58667, 0.47733, 0.56267, 0.56000, 0.52533, 0.56800, 0.53333, 0.61867, 0.62133, 0.57867, 0.54400, 0.57067, 0.55200, 0.62133, 0.64267, 0.50933, 0.58133, 0.60533, 0.55200, 0.58400, 0.59733, 0.48533, 0.54933, 0.57067, 0.57333, 0.61333, 0.52533, 0.64267, 0.59733, 0.56800, 0.54667, 0.49067, 0.55200, 0.59467, 0.57867, 0.55733, 0.58400, 0.58400, 0.48267, 0.62400, 0.57067, 0.62133, 0.56533, 0.56800, 0.55467, 0.57600, 0.54667, 0.62133, 0.56800, 0.51733, 0.57067, 0.61600, 0.57333, 0.66133, 0.50933, 0.58133, 0.60267, 0.52533, 0.60000, 0.63467, 0.52800, 0.55733, 0.58667, 0.58667, 0.61867, 0.56800, 0.54400, 0.59467, 0.57067, 0.56533, 0.60533, 0.55467, 0.63733, 0.58933, 0.59733, 0.59733, 0.59200, 0.56000, 0.60800, 0.57867, 0.59467, 0.58400, 0.53867, 0.50933, 0.59200, 0.57600, 0.57333, 0.63733, 0.52800, 0.56800, 0.56267, 0.65333, 0.62667, 0.51733, 0.54400, 0.51467, 0.56533, 0.53867, 0.50933, 0.53333, 0.54933, 0.58400, 0.60000, 0.56533, 0.54400, 0.52267, 0.59467, 0.58133, 0.56800, 0.53333, 0.60267, 0.54400, 0.56533, 0.59467, 0.57333, 
accuracy: 0.571933333333 sem: 0.00687127231147
