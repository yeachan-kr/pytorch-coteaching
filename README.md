# pytorch-coteaching
Pytorch implementations fo Co-teaching for noisy label learning



### Experiments on CIFAR-10

#### Performance results
| Settings / Models   	| CNN (reproduce, standard) 	| CNN (paper, standard) 	| CNN (reproduce, coteaching) 	| CNN (paper, coteaching) 	|
|---------------------	|:-------------------------:	|:---------------------:	|:---------------------------:	|:-----------------------:	|
| Clean (ε = 0%)      	|                           	|                       	|                             	|                         	|
| Sym (ε = 20%) 	|                           	|                       	|                             	|                         	|
| Sym (ε = 50%) 	|       43.0% (66.9%)       	|         48.87%        	|            72.2%            	|          74.02          	|


#### Learning curve 

<img src="./coteach_learning_curve.png" width="45%"> <img src="./normal_learning_curve.png" width="45%">
