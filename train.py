import sys
import random
import torch
from torch import nn
from torch.utils.data import DataLoader
import dataset
from model import ESMM
import numpy as np
from sklearn.metrics import roc_auc_score
import pdb


# super parameter
batch_size = 2000
embedding_size = 18
learning_rate = 1e-3
total_epoch = 10
earlystop_epoch = 1

vocabulary_size = {
    '101': 238635,
    '121': 98,
    '122': 14,
    '124': 3,
    '125': 8,
    '126': 4,
    '127': 4,
    '128': 3,
    '129': 5,
    '205': 467298,
    '206': 6929,
    '207': 263942,
    '216': 106399,
    '508': 5888,
    '509': 104830,
    '702': 51878,
    '853': 37148,
    '301': 4
}

model_file = '/home/wutongzhou/file/esmm-myself/out/ESMM.model'


def get_dataloader(filename, batch_size, shuffle):
  data = dataset.XDataset(filename)
  loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
  return loader


def train(train_dataloader, dev_dataloader, model, device, optimizer):
  model.to(device)# 
  best_acc = 0.0
  earystop_count = 0
  best_epoch = 0
  for epoch in range(total_epoch):
    total_loss = 0.
    nb_sample = 0
    # train
    model.train()
    for step, batch in enumerate(train_dataloader):
      click, conversion, features = batch
      for key in features.keys():
        features[key] = features[key].to(device)#
      click_pred, conversion_pred = model(features)
      
      loss = model.loss(click.float(),
                        click_pred,
                        conversion.float(),
                        conversion_pred,
                        device=device)
      optimizer.zero_grad()
      loss.sum().backward()
      optimizer.step()
      total_loss += loss.cpu().detach().numpy()
      nb_sample += click.shape[0]
      if step % 200 == 0:
        print('[%d] Train loss on step %d: %.6f' %
              (nb_sample, (step + 1), total_loss / (step + 1)))

    # validation
    print("start validation...")
    click_pred = []
    click_label = []
    conversion_pred = []
    conversion_label = []
    model.eval()
    for step, batch in enumerate(dev_dataloader):
      click, conversion, features = batch
      for key in features.keys():
        features[key] = features[key].to(device)

      with torch.no_grad():
        click_prob, conversion_prob = model(features)

      click_pred.append(click_prob.cpu())
      conversion_pred.append(conversion_prob.cpu())

      click_label.append(click)
      conversion_label.append(conversion)

    click_auc = cal_auc(click_label, click_pred)
    conversion_auc = cal_auc(conversion_label, conversion_pred)
    print("Epoch: {} click_auc: {} conversion_auc: {}".format(
        epoch + 1, click_auc, conversion_auc))

    acc = click_auc + conversion_auc
    if best_acc < acc:
      best_acc = acc
      best_epoch = epoch + 1
      torch.save(model.state_dict(), model_file)
      earystop_count = 0
    else:
      print("train stop at Epoch %d based on the base validation Epoch %d" %
            (epoch + 1, best_epoch))
      return


def prediction(train_dataloader,device):
  print("Start Prediction ...")

  model = ESMM(vocabulary_size, embedding_size)
  model.load_state_dict(torch.load(model_file))
  model.to(device)

  model.eval()
  click_list = []
  conversion_list = []  
  click_pred_list = []
  conversion_pred_list = []
  feature_list = []


  # dropout_layer = nn.Dropout(p=0.5)
  for i, batch in enumerate(train_dataloader):
    if i % 1000:
      sys.stdout.write("Prediction step:{}\r".format(i))
      sys.stdout.flush()
    click, conversion, features = batch
    for key in features.keys():
      features[key] = features[key].to(device)
    # features = {
    #   key: dropout_layer(features[key].float()).long() for key in features.keys()
    # }
    with torch.no_grad():
      click_pred, conversion_pred = model(features, True)
    click_list.append(click)
    feature_list.append(features)
    conversion_list.append(conversion)
    click_pred_list.append(click_pred)
    conversion_pred_list.append(conversion_pred)

  click_pred_list = torch.cat(click_pred_list, 0)
  conversion_pred_list = torch.cat(conversion_pred_list, 0)
  click_list = torch.cat(click_list, 0)
  conversion_list = torch.cat(conversion_list, 0)
  features = {}
  for key in vocabulary_size.keys():
    tmp = [feat[key] for feat in feature_list]
    features[key] = torch.cat(tmp, dim=0)
  return click_pred_list,conversion_pred_list, features, click_list, conversion_list


def test():
  print("Start Test ...")
  test_loader = get_dataloader('/home/wutongzhou/file/data/ctr_cvr.test',
                               batch_size=batch_size,
                               shuffle=False)
  model = ESMM(vocabulary_size, embedding_size)
  model.load_state_dict(torch.load(model_file))
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)
  model.eval()
  click_list = []
  conversion_list = []  
  click_pred_list = []
  conversion_pred_list = []
  for i, batch in enumerate(test_loader):
    if i % 1000:
      sys.stdout.write("test step:{}\r".format(i))
      sys.stdout.flush()
    click, conversion, features = batch
    for key in features.keys():
      features[key] = features[key].to(device)
    with torch.no_grad():
      click_pred, conversion_pred = model(features)
    click_list.append(click)
    conversion_list.append(conversion)
    click_pred_list.append(click_pred)
    conversion_pred_list.append(conversion_pred)
  click_auc = cal_auc(click_list, click_pred_list)
  conversion_auc = cal_auc(conversion_list, conversion_pred_list)
  print("Test Resutt: click AUC: {} conversion AUC:{}".format(
      click_auc, conversion_auc))


def cal_auc(label: list, pred: list):
  label = torch.cat(label)
  pred = torch.cat(pred)
  label = label.detach().cpu().numpy()
  pred = pred.detach().cpu().numpy()
  auc = roc_auc_score(label, pred, labels=np.array([0.0, 1.0]))
  return auc


if __name__ == "__main__":
  train_dataloader = get_dataloader('/home/wutongzhou/file/data/ctr_cvr.train',
                                    batch_size,
                                    shuffle=True)
  dev_dataloader = get_dataloader('/home/wutongzhou/file/data/ctr_cvr.dev',
                                  batch_size,
                                  shuffle=True)
  model = ESMM(vocabulary_size, embedding_size)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  optimizer = torch.optim.Adam(model.parameters(),
                               lr=learning_rate,
                               weight_decay=1e-5)
  # for i in range(1):
  train(train_dataloader, dev_dataloader, model, device, optimizer)
  
  ctr_list = []
  cvr_list = []

  train_loader = get_dataloader('/home/wutongzhou/file/data/ctr_cvr.train',
                              batch_size=batch_size,
                              shuffle=False)

  for i in range(int(total_epoch/2)):
    ctr_pred, cvr_pred, feats, ctr_labels, cvr_labels = prediction(train_loader,device = torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    ctr_list.append(ctr_pred)
    cvr_list.append(cvr_pred)

  
    ctr_mat = torch.stack(ctr_list, 0)
    cvr_mat = torch.stack(cvr_list, 0) #（20，n)
    features = feats



    variances_ctr = ctr_mat.var(0)  # 计算方差并加入列表
    variances_cvr = cvr_mat.var(0)

    keep = 0.1
    length_ctr = int(keep * ctr_mat.shape[1])  # 保留的方差的数量
    sorted_vars_ctr = np.argsort(variances_ctr.cpu().detach().numpy())[:length_ctr]
    sorted_vars_cvr = np.argsort(variances_cvr.cpu().detach().numpy())[:length_ctr]

    ctr_labels[sorted_vars_ctr] = ctr_mat.cpu().mean(0)[sorted_vars_ctr].to(torch.float64)
    cvr_labels[sorted_vars_cvr] = ctr_mat.cpu().mean(0)[sorted_vars_cvr].to(torch.float64)
    
    data = dataset.DRDataset(ctr_labels, cvr_labels, features, feature_names_path='/home/wutongzhou/file/data/ctr_cvr.train') 
    train_dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    # dim = 0 variance 
    # min, argmin 
    # min_indices = torch.argmin()
    # selected_samples = test_set[min_indices]

    test()
