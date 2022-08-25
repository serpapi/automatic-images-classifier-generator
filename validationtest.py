from dataset import CustomImageDataLoader, CustomImageDataset
from add_couchbase import ImagesDataBase
from commands import ValidationTestCommands
from models import *
import torch.nn.functional as F
import torch.nn as nn
import torch
import random
import numpy as np

class ValidationTest:
  def __init__(self, tc: ValidationTestCommands, cid: CustomImageDataset, db: ImagesDataBase, model: nn.Module):
    db = db()
    self.cid = cid(tc = tc, db = db)

    if "ids" in tc.dict() and tc.ids != None and tc.ids != []:
      self.ids = tc.ids
    elif "label_names" in tc.dict() and tc.label_names != None and tc.label_names != []:
      self.label_names = tc.label_names
      self.label_ids = []
      for label in self.label_names:
        returned_ids = db.get_image_keys_by_classification(label)
        for dict in returned_ids:
          self.label_ids.append(dict['id'])

    if "limit" in tc.dict() and tc.limit != None and tc.limit != 0:
      new_label_ids = []
      for i in range(0,tc.limit):
        random_pop_int = random.randint( 0, len(self.label_ids)-1 )
        new_label_ids.append(self.label_ids.pop(random_pop_int))
      self.label_ids = new_label_ids

    self.accuracy = 0.0
    self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    self.model = model(tc)
    self.model.load_state_dict(torch.load("models/{}.pt".format(tc.model_name)))
    self.model.eval()
  
  def test_accuracy(self):
    for id in self.label_ids:
      img, label = self.cid.get_item_by_id(id)

      img = [img.numpy()]
      img = np.asarray(img, dtype='float64')
      img = torch.from_numpy(img).float()
      img = img.to(self.device)

      label = [label.numpy()]
      label = np.asarray(label, dtype='float64')
      label = torch.from_numpy(label).float()
      label = label.to(self.device)

      if torch.cuda.is_available():
        self.model.cuda()
        prediction = self.model(img).to(self.device)[0]
      else:
        prediction = self.model(img)[0]


      if (label.argmax() == prediction.argmax()).item():
        self.accuracy = self.accuracy + (1.0/float(len(self.label_ids)))

    return self.accuracy