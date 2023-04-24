import logging
from datetime import datetime
import subprocess
import pandas as pd
import json
import shutil
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score

logging.basicConfig(filename="log.txt", level=logging.DEBUG,
                    format="%(asctime)s %(message)s", filemode="a")

# subprocess.run(["git", "clone", "https://huggingface.co/owen198/esgBERT_CICD"]) # first time
subprocess.run(["git", "fetch", "origin", "main"], cwd="/workspace/Step3/esgBERT_dataset")
subprocess.run(["git", "reset", "--hard", "FETCH_HEAD"], cwd="/workspace/Step3/esgBERT_dataset")
subprocess.run(["git", "fetch", "origin", "main"], cwd="/workspace/Step3/esgBERT_CICD")
subprocess.run(["git", "reset", "--hard", "FETCH_HEAD"], cwd="/workspace/Step3/esgBERT_CICD")

train_df = pd.read_csv('esgBERT_dataset/train.csv')
used_set = set(pd.read_csv('train_used.csv').text)

train_length = len(train_df)
train_df = train_df[~train_df.text.isin(used_set)].reset_index().drop(columns=['index'])
new_data_length = len(train_df)
print(f'train.csv new table rows: {new_data_length} / {train_length}')

train_labels = train_df['label'].tolist()
train_texts = train_df['text'].tolist()

dev_df = pd.read_csv('esgBERT_dataset/dev.csv')

dev_labels = dev_df['label'].tolist()
dev_texts = dev_df['text'].tolist()

timestamp = datetime.fromtimestamp(datetime.timestamp(datetime.now())).strftime("%Y/%m/%d, %H:%M:%S")
if new_data_length == 0:
    logging_df = pd.DataFrame([[timestamp, False, False, new_data_length, train_length, None, None, None]],columns=['Timestamp', 'Did Train', 'Model Update', 'Unseen Data', 'All Data', 'Accuracy', 'F1 Weighted', 'Loss'])
    logging_df.to_csv('metric_logs/logs.csv', mode='a', header=False, index=False)
    exit()

logging.info("New Training")
logging.info(f"trained data count: {len(used_set)}")
logging.info(f"new data count(not_used/total): {new_data_length} / {train_length}")

labels = ['Access to Communications', 'Access to Finance', 'Access to Health Care', 'Accounting', 'Biodiversity and Land Use', 'Board', 'Business Ethics', 'Carbon Emissions', 'Chemical Safety', 'Climate Change Vulnerability', 'Community Relations', 'Controversial Sourcing', 'Electronic Waste', 'Financial Product Safety', 'Financing Environmental Impact', 'Health and Demographic Risk', 'Health and Safety', 'Human Capital Development', 'Labor Management', 'Opportunities in Clean Tech', 'Opportunities in Green Building', 'Opportunities in Nutrition and Health', 'Opportunities in Renewable Energy', 'Ownership and Control', 'Packaging Material and Waste', 'Pay', 'Privacy and Data Security', 'Product Carbon Footprint', 'Product Safety and Quality', 'Raw Material Sourcing', 'Responsible Investment', 'Supply Chain Labor Standards', 'Tax Transparency', 'Toxic Emissions and Waste', 'Water Stress']
label2Int = {labels[i]: i for i in range(len(labels))}
int2Label = {i: labels[i] for i in range(len(labels))}
train_labels = [label2Int[label] for label in train_labels]
dev_labels = [label2Int[label] for label in dev_labels]

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
train_data = tokenizer(train_texts, padding=True, truncation=True, return_tensors="pt")
dev_data = tokenizer(dev_texts, padding=True, truncation=True, return_tensors="pt")

train_data['labels'] = train_labels
dev_data['labels'] = dev_labels


training_dataset = Dataset.from_dict(train_data)
dev_dataset = Dataset.from_dict(dev_data)

tokenized_datasets = DatasetDict(
    {
        "train": training_dataset,
        "validation": dev_dataset
    }
)

checkpoint = "esgBERT_CICD/esgBERT"
model = BertForSequenceClassification.from_pretrained(checkpoint, num_labels=35, id2label=int2Label, label2id=label2Int)


training_args = TrainingArguments(
    "./checkpoints",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=100,
    learning_rate=5e-7,
    save_total_limit = 2,
    load_best_model_at_end=True,
    weight_decay=0.01,
)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    metrics = {"accuracy": accuracy_score(labels, preds)}
    for average in ['micro', 'macro', 'weighted']:
        metrics[f'f1_{average}'] = f1_score(labels, preds, average=average)
    return metrics


trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics,
)
print(trainer.train())
print(trainer.evaluate())

shutil.copytree("checkpoints/runs", "esgBERT_CICD/logs", dirs_exist_ok=True)


prev_metrics = {}
with open("esgBERT_CICD/metrics.json", "r") as file:
    prev_metrics = json.load(file)

metrics = trainer.evaluate()
logging.info(f"new model metrics: {metrics}")

logging_df = pd.DataFrame([[timestamp, True, True, new_data_length, train_length, metrics['eval_accuracy'], metrics['eval_f1_weighted'], metrics['eval_loss']]],columns=['Timestamp', 'Did Train', 'Model Update', 'Unseen Data', 'All Data', 'Accuracy', 'F1 Weighted', 'Loss'])
    
try:
    if metrics['eval_loss'] <= prev_metrics['eval_loss'] and metrics['eval_f1_weighted'] >= prev_metrics['eval_f1_weighted']:
        with open("esgBERT_CICD/metrics.json", "w") as file:
            json.dump(metrics, file)
        trainer.save_model(checkpoint)
        print("model updated")
        logging.info(f"model updated.")
        shutil.copytree("checkpoints/runs", "esgBERT_CICD/logs", dirs_exist_ok=True)
        print(subprocess.run(["git", "add", "esgBERT", "metrics.json", "logs/"], cwd="/workspace/Step3/esgBERT_CICD"))
        print(subprocess.run(["git", "commit", "-m", f"bot: model update"], cwd="/workspace/Step3/esgBERT_CICD"))
        print(subprocess.run(["git", "push"], cwd="/workspace/Step3/esgBERT_CICD"))
        print("model pushed to Hugging Face hub")
        logging.info(f"model pushed to Hugging Face hub")
        shutil.copyfile("esgBERT_dataset/train.csv", "train_used.csv")
        print("'train_used.csv' is updated")
        logging.info("'train_used.csv' is updated")
    else:
        logging_df.loc[0, ['Is Better']] = [False]
        print("model is not better after training, model won't be saved.")
        logging.info(f"metrics is not better, won't update model.")
except KeyError:
    print("metrics.json caused KeyError.")
finally:
    timestamp = datetime.fromtimestamp(datetime.timestamp(datetime.now())).strftime("%Y/%m/%d, %H:%M:%S")
    logging_df.to_csv('metric_logs/logs.csv', mode='a', header=False, index=False)
    print("logs.csv updated")

"""## run model
from transformers import BertForSequenceClassification, pipeline
model = BertForSequenceClassification.from_pretrained(checkpoint)
clf = pipeline("text-classification", model, tokenizer=tokenizer)
clf("There's a wild fire")
"""