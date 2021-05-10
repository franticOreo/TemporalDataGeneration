# TemporalDataGeneration
---
This repository aims to create randomly generated [Symbolic Time Sequences](https://jmlr.csail.mit.edu/papers/volume18/15-403/15-403.pdf) that are templated from a pattern object. The pattern object can be customised by different parameters including: 
* The amount of body nodes (```low_body``` and ```high_body```) 
* The head probablitity range (```low_prob``` and ```high_prob```)
* Creation of disjunction (e.g ```disjunction=True```)
* Creation of negation (e.g ```negation=True```)
* Creation of conjunction (e.g ```conjunction=True```)
* Creation of cycle (e.g ```cycle=True```)

The time sequence can be customised with the following parameters:
* The amount of events (```n_patterns```)
* The amount of noisy events (```n_subsets```)
* The amount of noisy symbols (```n_noisy_symbols```)
* The probability of the consequent occuring (```head_prob```)

After the pattern has been generated and relevent event and prediction sequences have been created, the script creates and modifies .XML files to be inputted into ```TITARL.exe```. TITARL is a "Data Mining algorithm that is able to extract Temporal Association Rules from Symbolic Time Sequences and Scalar Time Series"[[ref](http://mathieu.guillame-bert.com/titarl_tutorial)]. After running the algorithm, the script generates a modified .html for the pattern rules for easy reference. 

#### Example Usage
```python data_generation.py --low_body=4 --high_body=6 --low_prob=60 --high_prob=90 --disjunction=True --n_patterns=10000 --n_subsets=5000```
