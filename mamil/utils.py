import torch
from collections import Counter
from torch.utils.data.sampler import WeightedRandomSampler


def get_class_balancing_weights(samples):
    class_counts = Counter()
    labels = []
    for sample in samples:
        _patches, label = sample
        bag_label = label[0]
        class_counts[bag_label] += 1
        labels.append(bag_label)
    total = float(len(labels))
    class_weights = {
        class_name: total / float(class_count)
        for class_name, class_count in class_counts.items()
    }
    return [class_weights[label] for label in labels]


def init_weights(attention_model):
    def weights_init_normal(m):
        '''Takes in a module and initializes all linear layers with weight
           values taken from a normal distribution.'''

        classname = m.__class__.__name__
        # for every Linear layer in a model
        if isinstance(m, torch.nn.Linear):
            y = m.in_features
            # m.weight.data shoud be taken from a normal distribution
            # m.weight.data.normal_(0.0,1/np.sqrt(y))
            torch.nn.init.xavier_normal_(m.weight)
            # m.bias.data should be 0
            m.bias.data.fill_(0)
            print("Init linear")
        elif isinstance(m, torch.nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)
            # torch.nn.init.xavier_uniform(m.bias)
            print("Init conv")
        else:
            print(f"{type(m)} is not initialized")
    for seq in [attention_model.feature_extractor_part1, attention_model.feature_extractor_part2]:
        for layer in seq:
            weights_init_normal(layer)
