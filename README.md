# MLOps for HF Classification Models
![System Sequence Diagram](imgs/SSD.png "System Sequence Diagram")
## Description

`hf_dataset`: dataset repository hosted on Hugging Face Hub.
`hf_model`: model repository hosted on Hugging Face Hub.


`esgBERTv4.py`: training script \
`train_used.csv`: all used training data \
`webhook.py`: flask server to receive webhook trigger \
`metric_logs`: directory to store each run's metrics \
`log.txt`: log of the training process

## Instructions

### Requirements
* Git, Git LFS
* To install Python packages with pip: `$ pip install -r requirements.txt`
### Setup
1. Clone the model and dataset repo and set up the git credential.
2. Make sure that the working directory, file names are set correctly. (recommend to use SSH)
3. Setup the webhook flask server: `python webhook.py`
4. Set the webhook in Hugging Face: "Settings->Webhooks". It should be triggered by the `hf_dataset` repository's main branch commit.

### Flow
1. New dataset committed and pushed to `hf_dataset`.
2. Triggered webhook, training script started.
3. Train on unseen data, starting from the previous model. 
4. If the new model has better metrics: \
    a. Log model metric. \
    b. Commit and push new model and metric to the `hf_model` repository. \
    c. Update `train_used.csv`
5. If the new model DOESN'T have better metrics: do nothing

### Tips 
* Make sure to run the webhook server in the background. 
* Using ngrok is an option for testing.
* If the run failed, change "commit.txt" and replay the webhook to try again without a new commit.

## Future Works
* Host training dataset on a database
* Use Hugging Face Hub's library to handle pulling/pushing of the repos
* Implement with DevOps, CI/CD tools

