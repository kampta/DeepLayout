import torch
from torch import nn
import torch.nn.functional as F


class LabelSetEncoder(nn.Module):
    def __init__(self, number_labels):
        super(LabelSetEncoder, self).__init__()

        self.number_labels = number_labels
        self.fc1 = nn.Linear(self.number_labels, 128)
        self.fc2 = nn.Linear(128, 128)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x


class SingleLabelEncoder(nn.Module):
    def __init__(self, number_labels):
        super(SingleLabelEncoder, self).__init__()

        self.number_labels = number_labels
        self.fc1 = nn.Linear(self.number_labels, 128)
        self.fc2 = nn.Linear(128, 128)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x


class CountsEncoder(nn.Module):
    def __init__(self, number_labels):
        super(CountsEncoder, self).__init__()

        self.number_labels = number_labels
        self.fc1 = nn.Linear(self.number_labels, 128)
        self.fc2 = nn.Linear(128, 128)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x


class CountConditioningMLP(nn.Module):
    def __init__(self, number_labels):
        super(CountConditioningMLP, self).__init__()

        self.encode_label_set = LabelSetEncoder(number_labels)
        self.encode_single_label = SingleLabelEncoder(number_labels)
        self.encode_counts = CountsEncoder(number_labels)

        self.fc = nn.Linear(128 * 3, 128)

    # the LayoutVAE paper seems to indicate they teacher-force
    # at evaluation time... I can't imagine that is correct?
    def forward(self, label_set, current_label, count_so_far):
        label_set = self.encode_label_set(label_set)
        current_label = self.encode_single_label(current_label)
        count_so_far = self.encode_counts(count_so_far)

        aggregate = torch.cat((label_set, current_label, count_so_far), dim=-1)
        aggregate = self.fc(aggregate)

        return aggregate


class CountInputEncoder(nn.Module):
    def __init__(self):
        super(CountInputEncoder, self).__init__()

        self.fc1 = nn.Linear(1, 128)
        self.fc2 = nn.Linear(128, 128)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class AutoregressiveCountEncoder(nn.Module):
    def __init__(self, number_labels, conditioning_size, representation_size=32):
        super(AutoregressiveCountEncoder, self).__init__()

        self.number_labels = number_labels

        self.input_encoder = CountInputEncoder()
        self.conditioning = CountConditioningMLP(self.number_labels)

        self.fc = nn.Linear(128 + conditioning_size, representation_size)
        self.project_mu = nn.Linear(representation_size, representation_size)
        self.project_s = nn.Linear(representation_size, representation_size)

    # x is the count to be encoded.
    def forward(self, x, label_set, current_label, count_so_far):
        x = self.input_encoder(x)
        condition = self.conditioning(label_set, current_label, count_so_far)

        x = torch.cat((x, condition), dim=-1)
        x = F.relu(self.fc(x))

        mu = self.project_mu(x)
        s = self.project_s(x)

        return mu, s, condition


class AutoregressiveCountDecoder(nn.Module):
    def __init__(self, conditioning_size, representation_size=32):
        super(AutoregressiveCountDecoder, self).__init__()

        self.fc1 = nn.Linear(conditioning_size + representation_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.project = nn.Linear(64, 1)

    def forward(self, z, condition):
        x = torch.cat((z, condition), dim=-1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # note, we are returning log(lambda)
        x = self.project(x)

        return x
