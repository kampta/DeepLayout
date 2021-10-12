import json
import numpy as np
import torch
from torch.utils.data import Dataset


class BatchCollator(object):
    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        indexes = transposed_batch[0]
        targets = transposed_batch[1]

        return indexes, targets


class TargetLayout(object):
    def __init__(self, label_set, count, bbox, label, width, height, annotation_id, permutation, image_id):
        device = bbox.device if isinstance(bbox, torch.Tensor) else torch.device("cpu")

        self.label_set = torch.as_tensor(label_set, dtype=torch.float32, device=device)
        self.count = torch.as_tensor(count, dtype=torch.float32, device=device)
        self.bbox = torch.as_tensor(bbox, dtype=torch.float32, device=device)
        self.label = torch.as_tensor(label, dtype=torch.float32, device=device)
        self.width = width
        self.height = height
        self.annotation_id = torch.as_tensor(annotation_id, device=device)
        self.permutation = torch.as_tensor(permutation, device=device)
        self.image_id = image_id

    def to(self, device):
        result = TargetLayout(
            self.label_set.to(device),
            self.count.to(device),
            self.bbox.to(device),
            self.label.to(device),
            self.width,
            self.height,
            self.annotation_id.to(device),
            self.permutation.to(device),
            self.image_id)

        return result

    def __len__(self):
        return self.bbox.shape[0]


class LayoutDataset(Dataset):
    def __init__(self, annotations_path, max_length=128):
        super(LayoutDataset, self).__init__()
        self.max_length = max_length
        self.annotations_path = annotations_path

        # load annotations.
        with open(self.annotations_path, "r") as f:
            self.data = json.load(f)

        self.categories = {c["id"]: c for c in self.data["categories"]}
        self.number_labels = len(self.categories)
        print("label set size: {0}".format(self.number_labels))

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate([c["id"] for c in self.categories.values()])
        }

        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }

        self.image_to_annotations = {}
        for annotation in self.data["annotations"]:
            image_id = annotation["image_id"]

            if not (image_id in self.image_to_annotations):
                self.image_to_annotations[image_id] = []

            self.image_to_annotations[image_id].append(annotation)

        label_sets = []
        counts = []
        boxes = []
        labels = []
        annotation_ids = []
        widths = []
        heights = []
        image_ids = []
        permutations = []

        self.images = []
        self.annotations = []

        for image in self.data["images"]:
            image_id = image["id"]
            height, width = float(image["height"]), float(image["width"])

            if image_id not in self.image_to_annotations:
                continue

            annotations = self.image_to_annotations[image_id]

            if (self.max_length is not None) and (len(annotations) > self.max_length):
                annotations = annotations[:self.max_length]

            # hack.
            for i, annotation in enumerate(annotations):
                annotation["index"] = i

            # sort the annotations left to right with labels (smallest first).
            sorted_annotations = []
            for label_index in range(self.number_labels):
                category_id = self.contiguous_category_id_to_json_id[label_index + 1]
                annotations_of_label = [a for a in annotations if a["category_id"] == category_id]
                annotations_of_label = list(sorted(annotations_of_label, key=lambda a: a["bbox"][0]))
                sorted_annotations += annotations_of_label

            self.annotations.append(sorted_annotations)

            label_set = np.zeros((self.number_labels,)).astype(np.uint8)
            count = np.zeros((self.number_labels,)).astype(np.uint8)
            box = np.zeros((len(sorted_annotations), 4))
            label = np.zeros((len(sorted_annotations),))
            annotation_id = np.zeros((len(sorted_annotations),))

            for annotation_index, annotation in enumerate(sorted_annotations):
                contiguous_id = self.json_category_id_to_contiguous_id[annotation["category_id"]]
                label_set[contiguous_id - 1] = 1
                count[contiguous_id - 1] += 1
                x, y, w, h = annotation["bbox"]

                # a good question is if we should divide by the long edge only.
                box[annotation_index] = np.array([x / width, y / height, w / width, h / height])
                label[annotation_index] = contiguous_id
                annotation_id[annotation_index] = annotation["id"]

            permutation = np.array([a["index"] for a in sorted_annotations]).astype(np.int)

            label_sets.append(label_set)
            counts.append(count)
            boxes.append(box)
            labels.append(label)
            widths.append(width)
            heights.append(height)
            annotation_ids.append(annotation_id)
            image_ids.append(image_id)
            permutations.append(permutation)
            self.images.append(image)

        self.label_sets = np.stack(label_sets, axis=0)
        self.counts = np.stack(counts, axis=0)
        self.boxes = boxes
        self.labels = labels
        self.widths = widths
        self.heights = heights
        self.annotation_ids = annotation_ids
        self.image_ids = image_ids
        self.permutations = permutations

        print("{0} images retained".format(len(self)))

    def __len__(self):
        return self.counts.shape[0]

    def __getitem__(self, index):
        # image_data = self.images[index]

        label_set = torch.from_numpy(self.label_sets[index])
        count = torch.from_numpy(self.counts[index])
        box = torch.from_numpy(self.boxes[index])
        label = torch.from_numpy(self.labels[index])
        width = self.widths[index]
        height = self.heights[index]
        annotation_id = torch.from_numpy(self.annotation_ids[index])
        image_id = self.image_ids[index]
        permutation = self.permutations[index]

        target = TargetLayout(label_set, count, box, label, width, height, annotation_id, permutation, image_id)

        return index, target
