# Neural Constituent Parser
Constituency parsing aims to extract a constituency-based parse tree from a sentence that 
represents its syntactic structure according to a phrase structure grammar.

We use a 2 element span (i,j,k,l) (representing 2 child phrases in the phrase tree) and
an encoding of the rule to score the spans on the sentence. We maximize the log-likelihood
of the probability of a phrase tree given a sentence which is proportional to the exponential
of the sum of the scores for the spans in the tree.

We also bound the score function using a RELu6 unit.

## Usage
First make sure all the dependencies are installed
``` shell
$ pip3 install -r requirements.txt
```
To train a model run <kbd>train.py</kbd>.

Run the following to see the arguments that can be passed to train.py
``` shell
$ python3 train.py -h
```

