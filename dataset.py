from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class XDataset(Dataset):
  '''load csv data with feature name ad first row'''
  def __init__(self, datafile):
    super(XDataset, self).__init__()
    self.feature_names = []
    self.datafile = datafile
    self.data = []
    self._load_data()

  def _load_data(self):
    print("start load data from: {}".format(self.datafile))
    count = 0
    with open(self.datafile) as f:
      self.feature_names = f.readline().strip().split(',')[2:]
      for line in f:
        count += 1
        line = line.strip().split(',')
        line = [int(v) for v in line]
        self.data.append(line)
        if count > 50000:
          break
    print("load data from {} finished".format(self.datafile))

  def __len__(self, ):
    return len(self.data)

  def __getitem__(self, idx):
    line = self.data[idx]
    click = float(line[0])
    conversion = float(line[1])
    features = dict(zip(self.feature_names, line[2:]))
    return click, conversion, features


class DRDataset(Dataset):
  '''load csv data with feature name ad first row'''
  def __init__(self, click, conversion, features, feature_names_path):
    super(DRDataset, self).__init__()
    self.feature_names = []
    self.click_list = click
    self.conversion_list = conversion
    self.feature_list = features
    with open(feature_names_path) as f:
      self.feature_names = f.readline().strip().split(',')[2:]

  def __len__(self, ):
    return len(self.click_list)

  def __getitem__(self, idx):
    click = self.click_list[idx]
    conversion = self.conversion_list[idx]
    features = {}
    for name in self.feature_names:
      features[name] = self.feature_list[name][idx]
    return click, conversion, features
