# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Useful Resources
- [ScriptRunConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)
- [Configure and submit training runs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets)
- [HyperDriveConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py)
- [How to tune hyperparamters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)

## Summary
This data includes data about a previous marketing campaign completed by a bank. It includes customer data, such as age, education level, and which campaign the customer was a part of, and we will be using it to train our model to predict if a customer signed through the campaign.

The best performing model was the AutoML model. It had a 93% accuracy predicting if someone would sign up through the marketing campaign over HyperDrive.

## Scikit-learn Pipeline
The pipeline archetecture is pretty straightforward once we have the ML cluster set up and all necessary installs were imported. First, we created our parameter sampling with the RandomParameterSampler and implemented our Bandit Termination Policy. Then, we created an sklearn environment before creating a ScriptRunConfig to specify the config details of our training job. After that, we configured the HyperDrive to help identify the best hyperparameters for our predictions. Finally, we submitted the HyperDrive run and monitored the run. We were able to classify whether or not a customer would sign up through the marketing campaign with 91.87% certainty with our chosen hyperparameters (image below).

I chose two hyperparameters (the two listed in the train.py by default), --C and --max-iters. --C is a hyperparameter for stength of regularization and was used with loguniform. --max-iters was used to make sure that we didn't go over the max-iters for the model. 

I chose a Bandit Policy with a slack factor of 0.1. This will end the job when the primary metric isn't within 10% of the most successful job. Not only does this save me time as I look through the different runs, but it improves the computational efficiency of the cloud resources I'm using - thus saving me money.

![image.png](attachment:image.png) - I'm not sure if this attached properly so I've also added it as a file in this repo.

## AutoML
The AutoML hyperparameters had about a 93% accuracy rate, a little better than what HyperDrive came up with. I created a classification task and based its performance on an r2 squared value. 

## Pipeline comparison
AutoML was able to be a little more accurate than that of the HyperDrive. The SKlearn option was based on accuracy of the predictions and AutoML was based on the r2 squared so there was a little bit of difference there. AutoML was considerably easier to set up than the lines of code that it took for HyperDrive.

## Future work
I'd love to do additional work testing out some hyperparameters through HyperDrive. I felt adding another parameter or two would allow the HyperDrive to become a little more accurate. The same could be said for adding more parameters to AutoML.

Also, I'd like to dabble in Cloud Shell to gain additional knowledge on that front.

## Proof of cluster clean up!
![image-2.png](attachment:image-2.png) - I'm also not sure if this attached properly so I'm adding a screenshot to the repo.
