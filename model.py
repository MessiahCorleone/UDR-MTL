import torch
from torch import nn
import pdb


class Tower(nn.Module):
  def __init__(self,
               input_dim: int,
               dims=[200, 80, 32],
               drop_prob=[0.1, 0.3, 0.3]):
    super(Tower, self).__init__()
    self.dims = dims
    self.drop_prob = drop_prob
    self.layer = nn.Sequential(nn.Linear(input_dim, dims[0]), nn.ReLU(),
                               nn.Dropout(drop_prob[0]),
                               nn.Linear(dims[0], dims[1]), nn.ReLU(),
                               nn.Dropout(drop_prob[1]),
                               nn.Linear(dims[1], dims[2]), nn.ReLU(),
                               nn.Dropout(drop_prob[2]))

  def forward(self, x):
    x = torch.flatten(x, start_dim=1)
    x = self.layer(x)
    return x

class ESMM(nn.Module):
  def __init__(self,
               feature_vocabulary: dict(),
               embedding_size: int,
               tower_dims=[200, 80, 32],
               drop_prob=[0.1, 0.3, 0.3]):
    super(ESMM, self).__init__()
    self.feature_vocabulary = feature_vocabulary
    self.feature_names = sorted(list(feature_vocabulary.keys()))
    self.embedding_size = embedding_size 
    self.embedding_dict = nn.ModuleDict()
    self.__init_weight()

    self.tower_input_size = len(feature_vocabulary) * (embedding_size)
    self.click_tower = Tower(self.tower_input_size, tower_dims, drop_prob)
    self.conversion_tower = Tower(self.tower_input_size, tower_dims, drop_prob)

    self.info_layer = nn.Sequential(nn.Linear(tower_dims[-1], 32), nn.ReLU(),
                                    nn.Dropout(drop_prob[-1]))

    self.click_layer = nn.Sequential(nn.Linear(tower_dims[-1], 1),
                                     nn.Sigmoid())
                                     
    self.conversion_layer = nn.Sequential(nn.Linear(tower_dims[-1], 1),
                                          nn.Sigmoid())

    self.dropout_layer = nn.Dropout(p=0.5)



  def __init_weight(self, ):
    for name, size in self.feature_vocabulary.items():
      emb = nn.Embedding(size, self.embedding_size)
      nn.init.normal_(emb.weight, mean=0.0, std=0.01)
      self.embedding_dict[name] = emb

  def forward(self, x, test_dropout=False):
    feature_embedding = []
    for name in self.feature_names:
      embed = self.embedding_dict[name](x[name])
      if test_dropout:
        embed = self.dropout_layer(embed)
      feature_embedding.append(embed)
    feature_embedding = torch.cat(feature_embedding, 1)
    tower_click = self.click_tower(feature_embedding)

    tower_conversion = torch.unsqueeze(
        self.conversion_tower(feature_embedding), 1)

    info = torch.unsqueeze(self.info_layer(tower_click), 1)

    wtz = torch.sum(torch.cat([tower_conversion, info], 1), dim=1)

    click = torch.squeeze(self.click_layer(tower_click), dim=1)
    conversion = torch.squeeze(self.conversion_layer(wtz), dim=1)

    return click, conversion

  def loss(self,
           click_label,
           click_pred,
           conversion_label,
           conversion_pred,
           constraint_weight=0.6,
           device="gpu:1"):
    click_label = click_label.to(device)
    conversion_label = conversion_label.to(device)

    click_loss = nn.functional.binary_cross_entropy(click_pred, click_label)
    conversion_loss = nn.functional.binary_cross_entropy(
        conversion_pred, conversion_label,reduce=False)

    #IPS
    weights = click_label / (click_pred + 1e-8)
    weights = weights.detach()
    weights = torch.clamp(weights, 1e-5, 1e5)
    weighted_conversion_loss = weights * conversion_loss        
    weighted_conversion_loss = weighted_conversion_loss.mean()
    
    label_constraint = torch.maximum(conversion_pred - click_pred,
                                     torch.zeros_like(click_label))
    constraint_loss = torch.sum(label_constraint)

    loss = click_loss + weighted_conversion_loss + constraint_weight * constraint_loss
    return loss
