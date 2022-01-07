# large - MDP 
This is a library that implements Markov Decision Processes from dataset accompanied with a MDP solver which uses GPU optimized Value Iteration. 


## Installation

Directly from source:
```bash
git clone https://github.com/idigitopia/lmdp.git
cd lmdp
pip install -e .
```
## Reference:
Please use this bibtex if you would like to cite it:
```
@misc{magym,
      author = {Shrestha, Aayam},
      title = {lmdp: A MDP library to build and solve large MDPs very fast!},
      year = {2021},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/idigitopia/lmdp}},
    }
```

## Usage:
```python
import lmdp
```

## Testing:

- Install: ```pip install -e ".[test]" ```
- Run: ```pytest```
