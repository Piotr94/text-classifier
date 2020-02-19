# -*- coding: utf-8 -*-

import torch
import torchtext
import torch.nn.functional as F
from torch import optim, nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

TEXT = torchtext.data.Field(sequential=True, tokenize='spacy', lower=True, include_lengths=True)
LABEL = torchtext.data.Field(sequential=False, use_vocab=False)

headlines_dataset = torchtext.data.TabularDataset(
    path='./Graduate - HEADLINES dataset (2019-06).json', format='json', 
    fields={"headline": ("headline", TEXT), "is_sarcastic": ("is_sarcastic", LABEL)}
    )

train, test, valid = headlines_dataset.split([0.6, 0.2, 0.2], stratified=True, strata_field="is_sarcastic")
TEXT.build_vocab(train, vectors='glove.6B.100d', unk_init=torch.Tensor.normal_)

train_iter, test_iter, valid_iter = torchtext.data.BucketIterator.splits(
    (train, test, valid), sort_within_batch=True, 
    sort_key=lambda x: len(x.headline),
    batch_sizes=(32, 256, 256), device=device
)

class Net(nn.Module):
  def __init__(self, hidden_dim, num_layers):
    super(Net, self).__init__()
    emb_dim = 100
    self.hidden_dim = hidden_dim
    self.num_layers = num_layers
    vocab = TEXT.vocab
    
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
    self.embedding = nn.Embedding(len(vocab), emb_dim, padding_idx=PAD_IDX)
    self.embedding.weight.data.copy_(vocab.vectors)
    self.embedding.weight.data[UNK_IDX] = torch.zeros(emb_dim)
    self.embedding.weight.data[PAD_IDX] = torch.zeros(emb_dim)

    self.rnn = nn.LSTM(emb_dim, hidden_dim, num_layers, bidirectional=True, dropout=0.5)
    self.fc = nn.Linear(hidden_dim * 2, 1)
    self.sigmoid = nn.Sigmoid()

    self.dropout = nn.Dropout(0.5)

  def forward(self, x, text_lengths):

    embeddings = self.dropout(self.embedding(x))

    packed_embeddings = nn.utils.rnn.pack_padded_sequence(embeddings, text_lengths)
    
    output, (hidden, cell) = self.rnn(packed_embeddings)

    output, output_lengths = nn.utils.rnn.pad_packed_sequence(output)

    hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
    out = self.fc(hidden)
    return self.sigmoid(out)

hidden_dim = 256
num_layers = 2
net = Net(hidden_dim, num_layers)
net.to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

min_val_loss = float('inf')
epochs_wo_improve = 0
EARLY_STOPPING_PATIENCE = 5
model_path = "./headlines_classifier"

def count_accuracy(outputs, labels):
  return ((outputs>0.5).float() == labels.float()).float().sum() / len(outputs)

def evaluate_net(eval_iter):
  net.eval()
  loss = 0
  accuracy = 0
  with torch.no_grad():
    for i, data in enumerate(eval_iter, 1):
      text, text_lengths, labels = data.headline[0].to(device), data.headline[1].to(device), data.is_sarcastic.to(device)
      optimizer.zero_grad()
      outputs = net(text, text_lengths).squeeze()
      loss = criterion(outputs, labels.float())
      accuracy += count_accuracy(outputs, labels)
      loss += loss.item()
  return loss, accuracy

for epoch in range(50):
  print("\nEpoch %d" % (epoch+1,)) 
  net = net.train()
  training_loss = 0
  training_accuracy = 0
  for data in train_iter:
    text, text_lengths, labels = data.headline[0].to(device), data.headline[1].to(device), data.is_sarcastic.to(device)
    optimizer.zero_grad()
    outputs = net(text, text_lengths).squeeze()
    loss = criterion(outputs, (labels.float()+0.5)/2)
    training_accuracy += count_accuracy(outputs, labels)
    loss.backward()
    optimizer.step()
    training_loss += loss.item()
  print('training loss: %.3f' % (training_loss / len(train_iter),))
  print('training accuracy: %.3f' % (training_accuracy / len(train_iter),))
  
  val_loss, val_accuracy = evaluate_net(valid_iter)
  print('validation loss: %.3f' % (val_loss / len(valid_iter),))
  print('validation accuracy: %.3f' % (val_accuracy / len(valid_iter),))
  if val_loss < min_val_loss:
    print("model saved")
    torch.save(net, model_path)
    min_val_loss = val_loss
    epochs_wo_improve = 0
  else:
    epochs_wo_improve += 1
    if epochs_wo_improve == EARLY_STOPPING_PATIENCE:
      net = torch.load(model_path)
      print("\nEarly stopping")
      break


test_loss, test_accuracy = evaluate_net(test_iter)
print('\ntest loss: %.3f' % (test_loss / len(test_iter),))
print('test accuracy: %.3f' % (test_accuracy / len(test_iter),))