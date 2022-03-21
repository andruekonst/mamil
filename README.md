# MAMIL
Multi-Attention Multiple Instance Learning [[ArXiV](https://arxiv.org/abs/2112.06071)] method implementation.

## Abstract
MAMIL is a new multi-attention based method for solving the MIL problem, which takes into account the neighboring patches or instances of each analyzed patch in a bag.
In the method, one of the attention modules takes into account adjacent patches or instances, several attention modules are used to get a diverse feature representation of patches, and one attention module is used to unite different feature representations to provide an accurate classification of each patch (instance) and the whole bag.
A combined representation of patches and their neighbors embeddings is used.

## Prerequisites
Python 3.6+, PyTorch, Torchvision and Jupyter with matplotlib are required.

*Note that it is NOT recommended to install the dependencies in the root environment with pip package manager.*

The easiest way to install the dependencies is through [Conda](https://docs.conda.io/en/latest/) (or [Mamba](https://github.com/mamba-org/mamba)):
```
conda create -n mamil python=3.9
conda activate mamil
conda install numpy jupyter matplotlib
```
Install PyTorch distribution corresponding to hardware, following the [documentation](https://pytorch.org/get-started/locally/), e.g. for CUDA compatible linux machine:
```
conda install pytorch==1.9.0 torchvision=0.10.0 cudatoolkit=11.3 -c pytorch
```

Then setup the package:
```
python setup.py install
```
 

## Usage example

First, prepare a dataset and data loader for training.
In the following example we use MNIST dataset:

```{python}
from mamil.models import MAMIL1D
from mamil.utils import get_class_balancing_weights, make_test_plots
from mamil.datasets import MnistBags
from mamil.training import Args, train_model
import torch.utils.data as data_utils
from pathlib import Path


args = Args()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

train_weights = get_class_balancing_weights(train_set)
train_sampler = data_utils.WeightedRandomSampler(train_weights, len(train_weights))
train_shuffle = False  # no need to shuffle, because the sampler is used

loader_kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = data_utils.DataLoader(
    MnistBags(
        target_number=args.target_number,
        neighbour_number=args.neighbour_number,
        mean_bag_length=args.mean_bag_length,
        var_bag_length=args.var_bag_length,
        num_bag=args.num_bags_train,
        seed=args.seed,
        train=True
    ),
    batch_size=1,
    shuffle=train_shuffle,
    sampler=train_sampler,
    **loader_kwargs
)
```

Then choose an attention model depending on number of spatial dimensions,
optionally make checkpoints path and start training.
In the example we use 1D MAMIL model:

```{python}
model = MAMIL1D().double()
if args.cuda:
    model.cuda()

checkpoints_path = Path('checkpoints/') / f'{args.num_bags_train}__{args.seed}'
checkpoints_path.mkdir(exist_ok=True, parents=True)
model = train_model(
    model,
    args,
    train_loader,
    checkpoints_path=checkpoints_path
)
```

Prepare a new data loader for plots on the test dataset:

```{python}
plots_data_loader = data_utils.DataLoader(
    MnistBags(
        target_number=args.target_number,
        neighbour_number=args.neighbour_number,
        mean_bag_length=8,
        var_bag_length=0,
        num_bag=num_bags,
        seed=args.seed,
        train=False
    ),
    batch_size=1,
    shuffle=False,
    **loader_kwargs
)

make_test_plots(
    model,
    plots_data_loader,
    args,
    n_images=5,
    only=None,
    verbose=False,
    save_dir=Path('plots')
)
```

