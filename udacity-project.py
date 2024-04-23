#!/usr/bin/env python
# coding: utf-8

# In[16]:


from azureml.core import Workspace, Experiment

ws = Workspace.get(name = 'quick-starts-ws-258186', subscription_id = '5a4ab2ba-6c51-4805-8155-58759ad589d8', resource_group = 'aml-quickstarts-258186')

"""ws = Workspace.create(name='quick-starts-ws-258179',
            subscription_id='5a4ab2ba-6c51-4805-8155-58759ad589d8',
            resource_group='aml-quickstarts-258179',
            create_resource_group=True,
            location='eastus2'
            )"""
exp = Experiment(workspace=ws, name="udacity-project")

print('Workspace name: ' + ws.name, 
      'Azure region: ' + ws.location, 
      'Subscription id: ' + ws.subscription_id, 
      'Resource group: ' + ws.resource_group, sep = '\n')

run = exp.start_logging()


# In[17]:


from azureml.core.compute import ComputeTarget, AmlCompute

from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

cluster_name = "project-cpu"

# TODO: Create compute cluster
# Use vm_size = "Standard_D2_V2" in your provisioning configuration.
# max_nodes should be no greater than 4.

### YOUR CODE HERE ###
try:
    aml_compute = ComputeTarget(workspace=ws, name = cluster_name)
    print("Found existing cluster, use it")
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(vm_size="STANDARD_D2_V2", max_nodes= 4)
    aml_compute = ComputeTarget.create(ws, cluster_name, compute_config)


# In[44]:


from azureml.widgets import RunDetails
from azureml.train.sklearn import SKLearn
from azureml.train.hyperdrive.run import PrimaryMetricGoal
from azureml.train.hyperdrive.policy import BanditPolicy
from azureml.train.hyperdrive.sampling import RandomParameterSampling
from azureml.train.hyperdrive.runconfig import HyperDriveConfig
from azureml.train.hyperdrive.parameter_expressions import choice, loguniform
from azureml.core import Environment, ScriptRunConfig
import os

# Specify parameter sampler
ps = RandomParameterSampling(
{
    '--C': loguniform(-3,3),
    '--max_iter': choice(100, 200, 300, 400)
})

"""ps = RandomParameterSampling(
    {
        '--batch-size': choice(25,50,100),
        '--first-layer-neurons': choice(10, 50, 200, 300, 500),
        '--second-layer-neurons': choice(10, 50, 200, 500),
        '--learning-rate': loguniform(-6,-1)
    }
)"""

# Specify a Policy
policy = BanditPolicy(slack_factor = 0.1, delay_evaluation= 5, evaluation_interval=1)

if "training" not in os.listdir():
    os.mkdir("./training")

# Setup environment for your training run
sklearn_env = Environment.from_conda_specification(name='sklearn-env', file_path='conda_dependencies.yml')

# Create a ScriptRunConfig Object to specify the configuration details of your training job
src = ScriptRunConfig(
    source_directory= ".",
    script = "train.py",
    compute_target= cluster_name,
    environment = sklearn_env
)

# Create a HyperDriveConfig using the src object, hyperparameter sampler, and policy.
hyperdrive_config = HyperDriveConfig(
    run_config = src,
    hyperparameter_sampling=ps,
    policy= policy,
    primary_metric_name="Accuracy",
    primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
    max_total_runs=20,
    max_concurrent_runs=4
)


# In[45]:


# Submit your hyperdrive run to the experiment and show run details with the widget.

### YOUR CODE HERE ###
hdr = exp.submit(config = hyperdrive_config)

RunDetails(hdr).show()

notebook_run = exp.start_logging()

notebook_run.log(name="message", value = "Hello from run!")

print(notebook_run.get_status())


# In[65]:


import joblib
# Get your best run and save the model from that run.

### YOUR CODE HERE ###
best_run = hdr.get_best_run_by_primary_metric()
best_run_metrics = best_run.get_metrics()
print(best_run_metrics)

## SAVE THE BEST MODEL HERE ###


# In[ ]:


from azureml.data.dataset_factory import TabularDatasetFactory

# Create TabularDataset using TabularDatasetFactory
# Data is available at: 
# "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"

### YOUR CODE HERE ###

csv = 'https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv'

dataset = TabularDatasetFactory.from_delimited_files(csv)


# In[ ]:


from train import clean_data
from sklearn.model_selection import train_test_split 

# Use the clean_data function to clean your data.
x, y = clean_data(dataset)
    
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)


# In[ ]:


from azureml.train.automl import AutoMLConfig
import pandas as pd

# Set parameters for AutoMLConfig
# NOTE: DO NOT CHANGE THE experiment_timeout_minutes PARAMETER OR YOUR INSTANCE WILL TIME OUT.
# If you wish to run the experiment longer, you will need to run this notebook in your own
# Azure tenant, which will incur personal costs.
automl_config = AutoMLConfig(
    experiment_timeout_minutes=30,
    task='classification',
    primary_metric="r2_score",
    training_data=pd.concat([x_train, y_train],axis = 1),
    label_column_name="y",
    n_cross_validations=5)


# In[ ]:


# Submit your automl run

### YOUR CODE HERE ###
remote_run = exp.submit(automl_config, show_output = False)


# In[ ]:


# Retrieve and save your best automl model.

### YOUR CODE HERE ###
best_run_automl = remote_run.get_best_run_by_primary_metric()
best_run_metrics_automl = best_run.get_metrics()
print(best_run_metrics_automl)

