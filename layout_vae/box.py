import torch
from torch import nn
import torch.nn.functional as F

from count import LabelSetEncoder, SingleLabelEncoder


class BoxHistoryMemory(nn.Module):
    def __init__(self, number_labels):
        super(BoxHistoryMemory, self).__init__()

        self.memory = nn.LSTM(number_labels + 4, 128)

    def forward(self, labels, boxes, state=None):
        # I assume if there is no history, we just return the zero vector?
        device = labels.device
        batch_size = labels.size(0)
        history_length = labels.size(1)
        if history_length == 0:
            return torch.zeros((batch_size, 128)).to(device), None

        # make labels, and boxes into L x N
        input = torch.cat((labels, boxes), dim=-1).permute(1, 0, 2)

        output, state = self.memory(input, state)
        output = output[-1, :]

        return output, state


class BoxConditioningMLP(nn.Module):
    def __init__(self, number_labels):
        super(BoxConditioningMLP, self).__init__()

        self.encode_label_set = LabelSetEncoder(number_labels)
        self.encode_single_label = SingleLabelEncoder(number_labels)
        self.encode_box = BoxHistoryMemory(number_labels)

        self.fc = nn.Linear(128 * 3, 128)

    # the LayoutVAE paper seems to indicate they teacher-force
    # at evaluation time... I can't imagine that is correct?
    def forward(self, label_set, current_label, labels_so_far, boxes_so_far, state=None):
        label_set = self.encode_label_set(label_set)
        current_label = self.encode_single_label(current_label)
        boxes_so_far, state = self.encode_box(labels_so_far, boxes_so_far, state)

        aggregate = torch.cat((label_set, current_label, boxes_so_far), dim=-1)
        aggregate = self.fc(aggregate)

        return aggregate, state


class BoxInputEncoder(nn.Module):
    def __init__(self):
        super(BoxInputEncoder, self).__init__()

        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class AutoregressiveBoxEncoder(nn.Module):
    def __init__(self, number_labels, conditioning_size, representation_size=32):
        super(AutoregressiveBoxEncoder, self).__init__()

        self.number_labels = number_labels

        self.input_encoder = BoxInputEncoder()
        self.conditioning = BoxConditioningMLP(self.number_labels)

        self.fc = nn.Linear(128 + conditioning_size, representation_size)
        self.project_mu = nn.Linear(representation_size, representation_size)
        self.project_s = nn.Linear(representation_size, representation_size)

    # x is the count to be encoded.
    def forward(self, x, label_set, current_label, labels_so_far, boxes_so_far, state=None):
        x = self.input_encoder(x)
        condition, state = self.conditioning(label_set, current_label, labels_so_far, boxes_so_far, state=state)

        x = torch.cat((x, condition), dim=-1)
        x = F.relu(self.fc(x))

        mu = self.project_mu(x)
        s = self.project_s(x)

        return mu, s, condition, state


class AutoregressiveBoxDecoder(nn.Module):
    def __init__(self, conditioning_size, representation_size=32):
        super(AutoregressiveBoxDecoder, self).__init__()

        self.fc1 = nn.Linear(conditioning_size + representation_size, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 512)
        self.project = nn.Linear(512, 4)

        self.actvn = nn.LeakyReLU(0.2, False)

    def forward(self, z, condition):
        x = torch.cat((z, condition), dim=-1)

        x = self.actvn(self.fc1(x))
        x = self.actvn(self.fc2(x))
        x = self.actvn(self.fc3(x))
        x = F.sigmoid(self.project(x))

        return x
