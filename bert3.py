# import torch.autocast
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from transformers import DistilBertModel, DistilBertTokenizer, DistilBertForSequenceClassification, TrainerCallback
from transformers import TrainingArguments, Trainer
from transformers import EvalPrediction
import torch

# Took inspiration from https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb#scrollTo=797b2WHJqUgZ

# Returns a list of Duke Class corresponding Cornell Class label
def get_labels(cornell_df, duke_df):
    res = np.zeros(34)
    i=0
    for row in duke_df['Cornell Class Code']:
      res[i]= cornell_df[cornell_df['title'] == row]['ID'].values.item()
      i=i+1
    return res
# Make a mapping in each direction from id <=> label
def make_mapping(labels):
  iToL = {id:label for id, label in enumerate(labels)}
  lToI = {label:id for id, label in enumerate(labels)}
  return iToL, lToI

# Prepare a dataframe of Cornell data
def prep():
    filtered = pd.read_csv('./stringcourse.csv',  sep=',',  header=0)
    filtered['ID'] = range(0, len(filtered))
    filtered.columns = ['text', 'ID']
    filtered['title'] = filtered['text'].apply(lambda x: str(
        x).split()[1].upper() +" "+ str(x).split()[2] if len(str(x).split()) > 1 else None)
    filtered.columns = ['text', 'ID', 'title']
    return filtered

# Get a the embeddings
def get_embeddings(ds):
    # Inputs of dataset
    text = ds['inputs']
    # Label matrix zeroed out
    labels_matrix = np.zeros((len(text), len(labels)))
    # Tokenize our text
    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=128)
    for id, label in enumerate(labels):
      labels_matrix[:, id] = label
    # now we have a label matrix in our encoding
    encoding["labels"] = labels_matrix.tolist()

    return encoding

# Prepare dataframe
cornell_df = prep()
# Get our input training descriptions
texts = cornell_df['text'].tolist()
# Create dataframe for Duke Classes
duke_df = pd.read_csv('Duke Roster.csv')
# Get corresponding labels for Cornell class (IDs in order)
labels = [i for i in range(1954)]
# Get labels for Duke classes
duke_labels = get_labels(cornell_df, duke_df)
# Create tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# Obtain Duke class descriptions
new_texts = duke_df['Duke Description'].tolist()
# Create a dataset for Cornell classes (training)
train_ds = Dataset.from_dict({"inputs": texts, "labels": labels})
# Create a dataset for Duke classes (testing)
eval_ds =  Dataset.from_dict({"inputs": new_texts, "labels": duke_labels})
# Tokenize the training dataset
encoded_train_ds = train_ds.map(get_embeddings, batched=True, remove_columns=train_ds.column_names)
# Tokenize the testing dataset
encoded_eval_ds = eval_ds.map(get_embeddings, batched=True, remove_columns=eval_ds.column_names)
# Create our mapping from id <=> label
iToL, lToI = make_mapping(labels)
# Instantiate our bert model
model = DistilBertForSequenceClassification.from_pretrained("bert-base-uncased",
# Put the model on the GPU                                                      num_labels=len(labels))
model.cuda()
# We are considering batches of 16
batch_size=16
# Our training metric is the hit rate
metric_name = "hit_rate"
# Set up our arguments. 
args = TrainingArguments(
    output_dir = "/contents/",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=0.01,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=2,
    load_best_model_at_end=True,
)

# A list of our predictions on the last iteration
final_pred = []
# Our hit rate in terms of obtaining a value in the top 5
def get_hit_rate(top5_values, labels):
  hit_rate = 0
  for i in range(len(labels)):
    for j in range(5):
      if(labels[i] == top5_values[j]):
        hit_rate +=1
  return hit_rate

# The rmse based on top prediction
def get_rmse(y_pred, labels):
  mse = torch.mean((y_pred - labels)**2)
  rmse = torch.sqrt(mse).item()
  return rmse

# The arhr based on the top prediction
def get_arhr(predictions, labels):
  _, ranks = torch.sort(predictions, descending=True, dim=1)
  _, target_rank = torch.sort(ranks, dim=1)

  reciprocal_rank = 1.0 / (target_rank + 1).float()
  arhr = torch.mean(reciprocal_rank)

  return arhr.item()

# This is buggy. Predictions is treated if it contains values (0,...,1953)
# But it contains an output which must be reformatted for our task

def metrics(predictions, labels):
  # Calculates hit rate
  hit_rate = get_hit_rate(predictions, labels)
  # Calculate rmse
  rmse = get_rmse(predictions, labels)
  # Calculate mae
  mae =  torch.mean(torch.abs(predictions - labels))
  # Calculate arhr
  arhr = get_arhr(predictions,labels)
  # Make a diction of these metrics
  metrics = { 'hit_rate' : hit_rate,
             'rmse':rmse,
             'mae':mae,
             'arhr':arhr
  }
  return metrics

# Takes the predictions and finds the proper metrics for it
def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions,
            tuple) else p.predictions
    result = metrics(
        predictions=preds,
        labels=p.label_ids)
    return result

# Create  a Callback Class which will give us training / evaluationi statistics as we go
class CustomCallback(TrainerCallback):
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer
    # At the end of each epoch, make note of metrics
    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy
# Initialize our trainer object
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=encoded_train_ds,
    eval_dataset=encoded_eval_ds,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)
trainer.add_callback(CustomCallback(trainer))
# Train the model
train = trainer.train()
# Get the metrics from training
metrics = train.metrics
# create a JSON to store the metrics
json_filename = '/content/train_metrics.json'  # Modify the path as needed

# open the json and append the metrics data to it
with open(json_filename, 'w') as json_file:
    json.dump(metrics, json_file)
# download the file
files.download(json_filename)

# Get the metrics from evaluation
metrics = trainer.evaluate()
# Make a filename
json_filename = '/content/eval_metrics.json'  # Modify the path as needed
# Append these metrics to the json file
with open(json_filename, 'w') as json_file:
    json.dump(metrics, json_file)
files.download(json_filename)


