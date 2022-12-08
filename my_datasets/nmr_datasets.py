import random
import torch
import numpy as np
from torch.utils.data import Dataset


class QEDDataset(Dataset):
    def __init__(self, **kwargs):
        super(QEDDataset, self).__init__()
        self.scale = kwargs.pop(
            'scale') if 'scale' in kwargs.keys() else 100
        self.min_value = kwargs.pop(
            'min_value') if 'min_value' in kwargs.keys() else 0
        self.max_value = kwargs.pop(
            'max_value') if 'max_value' in kwargs.keys() else 1
        self.augment_prob = kwargs.pop(
            'augment_prob') if 'augment_prob' in kwargs.keys() else 0
        # augment_range unit ±ppm
        self.augment_range = kwargs.pop(
            'augment_range') if 'augment_range' in kwargs.keys() else 0.05
        self.units = (self.max_value - self.min_value) * self.scale
        self.start = kwargs.pop(
            'start') if 'start' in kwargs.keys() else 191
        self.end = kwargs.pop(
            'end') if 'end' in kwargs.keys() else 192
        self.number_start = kwargs.pop(
            'number_start') if 'number_start' in kwargs.keys() else 200

    def get_value(self, value, augment_prob):
        if random.random() < augment_prob:
            return value
        else:
            return 0

    # def move_parallelled(self, QED):
    #     QED = list(set(QED))
    #     new_QED = []
    #     offset = (random.random() * 2 - 1) * self.augment_range
    #     for value in QED:
    #         if random.random() < self.augment_prob:
    #             new_QED.append(value + offset)
    #         else:
    #             new_QED.append(value)
    #     return self.fill_item(new_QED)

    def fill_item(self, QED, want_token=True):
        QED = list(set(QED))
        QED = [round((value - self.min_value) * self.scale) for value in QED]
        QED = [min(max(0, value), self.units-1) for value in QED]
        QED.sort()
        if want_token:
            QED = [191] + [value + self.number_start for value in QED] + [192]
            item = {"input_ids": QED,
                    "attention_mask": [1 for _ in range(len(QED))]}
            return item
        else:
            item = np.zeros(self.units)
            item[QED] = 1
            item = torch.from_numpy(item).to(torch.float32)
            item_ = {"input_ids": [191, 200, 192],
                     "attention_mask": [1, 1, 1]}
            return item, item_

    def augment_item(self, QED, want_token=True):
        QED = list(set(QED))
        offset = np.random.normal(
            loc=0.0, scale=self.augment_range/2, size=len(QED))
        offset = [min(max(-self.augment_range, i), self.augment_range)
                  for i in offset]
        offset = [self.get_value(i, self.augment_prob) for i in offset]
        new_QED = [QED_item + offset_item for QED_item,
                   offset_item in zip(QED, offset)]
        return self.fill_item(new_QED, want_token=want_token)



class molecular_weightDataset(Dataset):
    def __init__(self, **kwargs):
        super(molecular_weightDataset, self).__init__()
        self.scale = kwargs.pop(
            'scale') if 'scale' in kwargs.keys() else 1
        self.min_value = kwargs.pop(
            'min_value') if 'min_value' in kwargs.keys() else 0
        self.max_value = kwargs.pop(
            'max_value') if 'max_value' in kwargs.keys() else 1000
        self.augment_prob = kwargs.pop(
            'augment_prob') if 'augment_prob' in kwargs.keys() else 0
        # augment_range unit ±ppm
        self.augment_range = kwargs.pop(
            'augment_range') if 'augment_range' in kwargs.keys() else 50
        self.units = (self.max_value - self.min_value) * self.scale
        self.start = kwargs.pop(
            'start') if 'start' in kwargs.keys() else 185
        self.end = kwargs.pop(
            'end') if 'end' in kwargs.keys() else 186
        self.number_start = kwargs.pop(
            'number_start') if 'number_start' in kwargs.keys() else 300

    def get_value(self, value, augment_prob):
        if random.random() < augment_prob:
            return value
        else:
            return 0

    # def move_parallelled(self, molecular_weight):
    #     molecular_weight = list(set(molecular_weight))
    #     new_molecular_weight = []
    #     offset = (random.random() * 2 - 1) * self.augment_range
    #     for value in molecular_weight:
    #         if random.random() < self.augment_prob:
    #             new_molecular_weight.append(value + offset)
    #         else:
    #             new_molecular_weight.append(value)
    #     return self.fill_item(new_molecular_weight)

    def fill_item(self, molecular_weight, want_token=True):
        molecular_weight = list(set(molecular_weight))
        molecular_weight = [round((value - self.min_value) * self.scale) for value in molecular_weight]
        molecular_weight = [min(max(0, value), self.units-1) for value in molecular_weight]
        molecular_weight.sort()
        if want_token:
            molecular_weight = [185] + [value + self.number_start for value in molecular_weight] + [186]
            item = {"input_ids": molecular_weight,
                    "attention_mask": [1 for _ in range(len(molecular_weight))]}
            return item
        else:
            item = np.zeros(self.units)
            item[molecular_weight] = 1
            item = torch.from_numpy(item).to(torch.float32)
            item_ = {"input_ids": [185, 300, 186],
                     "attention_mask": [1, 1, 1]}
            return item, item_

    def augment_item(self, molecular_weight, want_token=True):
        molecular_weight = list(set(molecular_weight))
        offset = np.random.normal(
            loc=0.0, scale=self.augment_range/2, size=len(molecular_weight))
        offset = [min(max(-self.augment_range, i), self.augment_range)
                  for i in offset]
        offset = [self.get_value(i, self.augment_prob) for i in offset]
        new_molecular_weight = [molecular_weight_item + offset_item for molecular_weight_item,
                   offset_item in zip(molecular_weight, offset)]
        return self.fill_item(new_molecular_weight, want_token=want_token)



class logPDataset(Dataset):
    def __init__(self, **kwargs):
        super(logPDataset, self).__init__()
        self.scale = kwargs.pop(
            'scale') if 'scale' in kwargs.keys() else 10
        self.min_value = kwargs.pop(
            'min_value') if 'min_value' in kwargs.keys() else -4
        self.max_value = kwargs.pop(
            'max_value') if 'max_value' in kwargs.keys() else 7
        self.augment_prob = kwargs.pop(
            'augment_prob') if 'augment_prob' in kwargs.keys() else 0
        # augment_range unit ±ppm
        self.augment_range = kwargs.pop(
            'augment_range') if 'augment_range' in kwargs.keys() else 0.5
        self.units = (self.max_value - self.min_value) * self.scale
        self.start = kwargs.pop(
            'start') if 'start' in kwargs.keys() else 193
        self.end = kwargs.pop(
            'end') if 'end' in kwargs.keys() else 194
        self.number_start = kwargs.pop(
            'number_start') if 'number_start' in kwargs.keys() else 1300

    def get_value(self, value, augment_prob):
        if random.random() < augment_prob:
            return value
        else:
            return 0

    # def move_parallelled(self, logP):
    #     logP = list(set(logP))
    #     new_logP = []
    #     offset = (random.random() * 2 - 1) * self.augment_range
    #     for value in logP:
    #         if random.random() < self.augment_prob:
    #             new_logP.append(value + offset)
    #         else:
    #             new_logP.append(value)
    #     return self.fill_item(new_logP)

    def fill_item(self, logP, want_token=True):
        logP = list(set(logP))
        logP = [round((value - self.min_value) * self.scale) for value in logP]
        logP = [min(max(0, value), self.units-1) for value in logP]
        logP.sort()
        if want_token:
            logP = [193] + [value + self.number_start for value in logP] + [194]
            item = {"input_ids": logP,
                    "attention_mask": [1 for _ in range(len(logP))]}
            return item
        else:
            item = np.zeros(self.units)
            item[logP] = 1
            item = torch.from_numpy(item).to(torch.float32)
            item_ = {"input_ids": [193, 1300, 194],
                     "attention_mask": [1, 1, 1]}
            return item, item_

    def augment_item(self, logP, want_token=True):
        logP = list(set(logP))
        offset = np.random.normal(
            loc=0.0, scale=self.augment_range/2, size=len(logP))
        offset = [min(max(-self.augment_range, i), self.augment_range)
                  for i in offset]
        offset = [self.get_value(i, self.augment_prob) for i in offset]
        new_logP = [logP_item + offset_item for logP_item,
                   offset_item in zip(logP, offset)]
        return self.fill_item(new_logP, want_token=want_token)



class SADataset(Dataset):
    def __init__(self, **kwargs):
        super(SADataset, self).__init__()
        self.scale = kwargs.pop(
            'scale') if 'scale' in kwargs.keys() else 10
        self.min_value = kwargs.pop(
            'min_value') if 'min_value' in kwargs.keys() else 1
        self.max_value = kwargs.pop(
            'max_value') if 'max_value' in kwargs.keys() else 10
        self.augment_prob = kwargs.pop(
            'augment_prob') if 'augment_prob' in kwargs.keys() else 0
        # augment_range unit ±ppm
        self.augment_range = kwargs.pop(
            'augment_range') if 'augment_range' in kwargs.keys() else 0.5
        self.units = (self.max_value - self.min_value) * self.scale
        self.start = kwargs.pop(
            'start') if 'start' in kwargs.keys() else 195
        self.end = kwargs.pop(
            'end') if 'end' in kwargs.keys() else 196
        self.number_start = kwargs.pop(
            'number_start') if 'number_start' in kwargs.keys() else 1410

    def get_value(self, value, augment_prob):
        if random.random() < augment_prob:
            return value
        else:
            return 0

    # def move_parallelled(self, SA):
    #     SA = list(set(SA))
    #     new_SA = []
    #     offset = (random.random() * 2 - 1) * self.augment_range
    #     for value in SA:
    #         if random.random() < self.augment_prob:
    #             new_SA.append(value + offset)
    #         else:
    #             new_SA.append(value)
    #     return self.fill_item(new_SA)

    def fill_item(self, SA, want_token=True):
        SA = list(set(SA))
        SA = [round((value - self.min_value) * self.scale) for value in SA]
        SA = [min(max(0, value), self.units-1) for value in SA]
        SA.sort()
        if want_token:
            SA = [195] + [value + self.number_start for value in SA] + [196]
            item = {"input_ids": SA,
                    "attention_mask": [1 for _ in range(len(SA))]}
            return item
        else:
            item = np.zeros(self.units)
            item[SA] = 1
            item = torch.from_numpy(item).to(torch.float32)
            item_ = {"input_ids": [195, 1410, 196],
                     "attention_mask": [1, 1, 1]}
            return item, item_

    def augment_item(self, SA, want_token=True):
        SA = list(set(SA))
        offset = np.random.normal(
            loc=0.0, scale=self.augment_range/2, size=len(SA))
        offset = [min(max(-self.augment_range, i), self.augment_range)
                  for i in offset]
        offset = [self.get_value(i, self.augment_prob) for i in offset]
        new_SA = [SA_item + offset_item for SA_item,
                   offset_item in zip(SA, offset)]
        return self.fill_item(new_SA, want_token=want_token)