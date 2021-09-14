# VChecker
An implementation of Zhang et. al's paper on 'Manually Detecting Errors for Data Cleaning Using Adaptive Crowdsourcing Strategies.'

To run, simply run 'python3 main.py pathtojson' from your console, where pathtojson is a file which contains a dictionary with two keys: *possible values*, denoting the data regions, and *difficulties*, denoting each region's difficulty.

e.g. {'possible_values':'red','blue', 'difficulties':'0.1','0.5'}

If the *pathtojson* argument is omitted, the default dataset *rainbow.json* is used.

The hyperparameters can be found in the func.py file.
