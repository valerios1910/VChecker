# VChecker
An implementation of Zhang et. al's paper on 'Manually Detecting Errors for Data Cleaning Using Adaptive Crowdsourcing Strategies.'

This implementation is for experimental purposes only, as real-world datasets and crowdsourcing are currently not supported. However, when necessary, they can be implemented with minimal effort. If anyone is interested, please drop me a message, or take the algorithm functions and plug them to your own code.

To execute the code, simply run 'python3 main.py pathtojson' from your console, where pathtojson points to a file which contains a dictionary with two keys: *possible values*, denoting the data regions, and *difficulties*, denoting each region's difficulty. The difficulties are only used for the crowdsourcing simulation and are not taken into account when executing the algorithms.

e.g. {'possible_values':'red','blue', 'difficulties':'0.1','0.5'}

If the *pathtojson* argument is omitted, the default dataset *rainbow.json* is used.

The hyperparameters can be found in the func.py file. Details can be found in the report pdf.
