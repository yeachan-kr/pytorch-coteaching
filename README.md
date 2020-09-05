# pytorch-coteaching
Pytorch implementations fo Co-teaching for noisy label learning



### Experiments on CIFAR-10
| Settings / Models   	| CNN (reproduce, standard) 	| CNN (paper, standard) 	| CNN (reproduce, coteaching) 	| CNN (paper, coteaching) 	|
|---------------------	|:-------------------------:	|:---------------------:	|:---------------------------:	|:-----------------------:	|
| Clean (ε = 0%)      	|                           	|                       	|                             	|                         	|
| Sym (ε = 20%) 	|                           	|                       	|                             	|                         	|
| Sym(ε = 20%) 	|       43.0% (66.9%)       	|         48.87%        	|            72.2%            	|          74.02          	|
