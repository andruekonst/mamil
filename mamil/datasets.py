"""Pytorch dataset object that loads MNIST dataset as bags."""

import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms


class MnistBags(data_utils.Dataset):
    def __init__(self, target_number=9, mean_bag_length=10, var_bag_length=2, num_bag=250, seed=1, train=True,
                 neighbour_number=None):
        self.target_number = target_number
        self.mean_bag_length = mean_bag_length
        self.var_bag_length = var_bag_length
        self.num_bag = num_bag
        self.train = train
        self.neighbour_number = neighbour_number

        self.r = np.random.RandomState(seed)

        self.num_in_train = 60000
        self.num_in_test = 10000

        if self.train:
            self.train_bags_list, self.train_labels_list = self._create_bags()
        else:
            self.test_bags_list, self.test_labels_list = self._create_bags()

    def _create_bags(self):
        if self.train:
            loader = data_utils.DataLoader(datasets.MNIST('../datasets',
                                                          train=True,
                                                          download=True,
                                                          transform=transforms.Compose([
                                                              transforms.ToTensor(),
                                                              transforms.Normalize((0.1307,), (0.3081,))])),
                                           batch_size=self.num_in_train,
                                           shuffle=False)
        else:
            loader = data_utils.DataLoader(datasets.MNIST('../datasets',
                                                          train=False,
                                                          download=True,
                                                          transform=transforms.Compose([
                                                              transforms.ToTensor(),
                                                              transforms.Normalize((0.1307,), (0.3081,))])),
                                           batch_size=self.num_in_test,
                                           shuffle=False)

        for (batch_data, batch_labels) in loader:
            all_imgs = batch_data
            all_labels = batch_labels

        bags_list = []
        labels_list = []

        for i in range(self.num_bag):
            bag_length = np.int32(self.r.normal(self.mean_bag_length, self.var_bag_length, 1))
            if bag_length < 1:
                bag_length = 1

            if self.train:
                indices = torch.LongTensor(self.r.randint(0, self.num_in_train, bag_length))
            else:
                indices = torch.LongTensor(self.r.randint(0, self.num_in_test, bag_length))

            labels_in_bag = all_labels[indices]
            if self.neighbour_number is None:
                labels_in_bag = labels_in_bag == self.target_number
            else:
                # labels_in_bag = (labels_in_bag == self.target_number) | (labels_in_bag == self.neighbour_number)
                labels_in_bag = 1 * (labels_in_bag == self.target_number) + 2 * (labels_in_bag == self.neighbour_number)

            bags_list.append(all_imgs[indices])
            labels_list.append(labels_in_bag)

        return bags_list, labels_list

    def __len__(self):
        if self.train:
            return len(self.train_labels_list)
        else:
            return len(self.test_labels_list)
    
    def _get_mil_label(self, labels_list):
        if self.neighbour_number is None:
            return max(labels_list)
        else:
            labels_arr = np.array(labels_list)
            # bag label is True if there are two consecutive True instance labels
            # return np.any(labels_arr[:-1] & labels_arr[1:])

            # bag label is True if there is at least one pair (main, neighbour) of consecutive numbers
            # for example:
            #   self.target_number == 9
            #   self.neighbour_number == 3
            #   number sequences [9, 3], [3, 9], [1, 3, 9, 2] are True
            #                    [9, 1, 3], [3, 1, 9] are False
            main_num_mask = labels_arr == 1  # here 1 is label of `target_number`
            nb_num_mask = labels_arr == 2  # here 2 is label of `neighbour_number`
            return np.any(main_num_mask[:-1] & nb_num_mask[1:]) or \
                   np.any(nb_num_mask[:-1] & main_num_mask[1:])
    
    def _get_instance_labels(self, labels_list):
        if self.neighbour_number is None:
            return labels_list
        else:
            # return list(map(lambda x: x != 0, labels_list))
            return np.array(labels_list) != 0

    def __getitem__(self, index):
        if self.train:
            bag = self.train_bags_list[index]
            label = [self._get_mil_label(self.train_labels_list[index]), self._get_instance_labels(self.train_labels_list[index])]
        else:
            bag = self.test_bags_list[index]
            label = [self._get_mil_label(self.test_labels_list[index]), self._get_instance_labels(self.test_labels_list[index])]

        return bag, label


def get_dataset_stats():
    dataset = datasets.MNIST(
        '../datasets',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ])
    )
    print("Dataset length:", len(dataset))
    bag, label = dataset[0]
    print("Label:", label)
    print("Bag shape:", bag.shape)

    common_bag = []
    for i, (bag, label) in zip(range(1000), dataset):
        common_bag.append(bag)
    common_bag = torch.cat(common_bag, dim=0).unsqueeze(1)
    mean = common_bag.mean(dim=(0, 2, 3))
    std = common_bag.std(dim=(0, 2, 3))
    print("Mean:", mean)
    print("Std:", std)


if __name__ == "__main__":

    train_loader = data_utils.DataLoader(MnistBags(target_number=9,
                                                   neighbour_number=3,
                                                   mean_bag_length=10,
                                                   var_bag_length=2,
                                                   num_bag=100,
                                                   seed=1,
                                                   train=True),
                                         batch_size=1,
                                         shuffle=True)

    test_loader = data_utils.DataLoader(MnistBags(target_number=9,
                                                  mean_bag_length=10,
                                                  var_bag_length=2,
                                                  num_bag=100,
                                                  seed=1,
                                                  train=False),
                                        batch_size=1,
                                        shuffle=False)

    len_bag_list_train = []
    mnist_bags_train = 0
    for batch_idx, (bag, label) in enumerate(train_loader):
        # bag_label = label[0].squeeze().item()
        # if not bag_label:
        #     continue
        # from matplotlib import pyplot as plt
        # instance_labels = label[1].squeeze().numpy()
        # fig, ax = plt.subplots(1, len(instance_labels))
        # for i in range(len(instance_labels)):
        #     ax[i].imshow(bag.squeeze()[i].numpy())
        #     ax[i].set_title(f'{instance_labels[i]}')
        # plt.show()
        # break
        len_bag_list_train.append(int(bag.squeeze(0).size()[0]))
        mnist_bags_train += label[0].numpy()[0]
    print('Number positive train bags: {}/{}\n'
          'Number of instances per bag, mean: {}, max: {}, min {}\n'.format(
        mnist_bags_train, len(train_loader),
        np.mean(len_bag_list_train), np.max(len_bag_list_train), np.min(len_bag_list_train)))

    len_bag_list_test = []
    mnist_bags_test = 0
    for batch_idx, (bag, label) in enumerate(test_loader):
        len_bag_list_test.append(int(bag.squeeze(0).size()[0]))
        mnist_bags_test += label[0].numpy()[0]
    print('Number positive test bags: {}/{}\n'
          'Number of instances per bag, mean: {}, max: {}, min {}\n'.format(
        mnist_bags_test, len(test_loader),
        np.mean(len_bag_list_test), np.max(len_bag_list_test), np.min(len_bag_list_test)))

    get_dataset_stats()
