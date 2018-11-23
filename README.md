# Uniform Semantic Role Labeling (Under Construction)

This repository contains code and models for replicating results from the following publication:
* [Jointly Predicting Predicates and Arguments in Neural Semantic Role Labeling](https://arxiv.org/abs/1805.04787)
* [Luheng He](https://homes.cs.washington.edu/~luheng), [Kenton Lee](http://kentonl.com/), [Omer Levy](https://levyomer.wordpress.com/) and [Luke Zettlemoyer](https://www.cs.washington.edu/people/faculty/lsz)
* In AAAI 2019

Part of the codebase is extended from [lsgn](https://github.com/luheng/lsgn). 

### Requirements
* Python 2.7
* TensorFlow 1.8.0
* pyhocon (for parsing the configurations)
* [tensorflow_hub](https://www.tensorflow.org/hub/) (for loading ELMo)

## Getting Started
* sudo apt-get install tcsh (Only required for processing CoNLL05 data)
* [GloVe](https://nlp.stanford.edu/projects/glove/) embeddings, the [srlconll](http://www.lsi.upc.edu/~srlconll/soft.html) scripts and the [eval09](http://ufal.mff.cuni.cz/conll2009-st/scorer.html) script:  
`./scripts/fetch_required_data.sh` 
* Build kernels: `./scripts/build_custom_kernels.sh` (Please make adjustments to the script according to your OS/gcc version)

## Setting up for ELMo (in progress)
* Some of our models are trained with the [ELMo embeddings](https://allennlp.org/elmo). We use the ELMo model loaded by [tensorflow_hub](https://www.tensorflow.org/hub/modules/google/elmo/2). You can download the file from [here]() and decompress it into the directory: `/elmo`.

## CoNLL Data
For replicating results on CoNLL-2005, CoNLL-2009 and CoNLL-2012 datasets, please follow the steps below.

### CoNLL-2005
The data is provided by:
[CoNLL-2005 Shared Task](http://www.lsi.upc.edu/~srlconll/soft.html),
but the original words are from the Penn Treebank dataset, which is not publicly available.
If you have the PTB corpus, you can run:  
` ./scripts/fetch_and_make_conll05_data.sh  /path/to/ptb/`  

### CoNLL-2009
The data is provided by:
[CoNLL-2009 Shared Task](http://ufal.mff.cuni.cz/conll2009-st/index.html),
Run:
`./scripts/make_conll2009_data.sh /path/to/conll-2009`

### CoNLL-2012
You have to follow the instructions below to get CoNLL-2012 data
[CoNLL-2012](http://cemantix.org/data/ontonotes.html), this would result in a directory called `/path/to/conll-formatted-ontonotes-5.0`.
Run:  
`./scripts/make_conll2012_data.sh /path/to/conll-formatted-ontonotes-5.0`

## Training Instructions

* Experiment configurations are found in `experiments.conf`
* Choose an experiment that you would like to run, e.g. `conll2012_best`
* For a single-machine experiment, run the following two commands:
  * `python singleton.py <experiment>`
  * `python evaluator.py <experiment>`
* Results are stored in the `logs` directory and can be viewed via TensorBoard.
* For final evaluation of the checkpoint with the maximum dev F1:
  * Run `python test_single.py <experiment>` for the single-model evaluation. For example: `python test_single.py conll2012_final`

## Other Quirks
* If you want to use GPUs, add command line parameter: `-gpu 0`.
* The evaluator should not be run on GPUs, since evaluating full documents does not fit within GPU memory constraints.
* The training runs indefinitely and needs to be terminated manually. The model generally converges at about 300k steps and within 12-36 hours.
* At test time, the code loads the entire GloVe 300D embedding file in the beginning, which would take a while.

