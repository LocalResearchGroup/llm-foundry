In this folder, please add YAML files for different hyperparameter sweeps that you want us to run. In this README file, put down ideas for different hyperparameters we should change and why. If it works, then we should try something else that's intuitive, and through that process something will be uncovered as a good set of hyperparameters. 


It seems like the most impactful hyperparameters to change are:
- learning rate
- LoRa adapter  config
- optimizers
- batch size
- dataset mixes


Look at the LRG meeting notes for the recommendations that Benjamin made about hyperparameter sweeps. I think he said we should use like 5% or 10% of the training data size and tokens to run the sweeps. 
