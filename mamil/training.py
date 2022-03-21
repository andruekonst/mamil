from pathlib import Path
import torch
from torch import nn
from torch.autograd import Variable
from typing import NamedTuple, Optional


class Args(NamedTuple):
    target_number: int = 9
    neighbour_number: int = 3
    mean_bag_length: int = 8
    var_bag_length: int = 2
    epochs: int = 20
    lr: float = 0.0005
    reg: float = 10e-5
    seed: int = 1
    no_cuda: bool = False
    cuda: bool = True
    model: str = 'template_attention'
    use_different_templates: bool = True
    use_positions: bool = False
    use_neighbourhood: bool = False
    n_templates: int = 10
    balance: bool = False
    num_bags_train: int = 100
    step_period: bool = 1


def init_weights(attention_model):
    def weights_init_normal(m):
        '''Takes in a module and initializes all linear layers with weight
           values taken from a normal distribution.'''

        classname = m.__class__.__name__
        # for every Linear layer in a model
        if isinstance(m, nn.Linear):
            y = m.in_features
            # m.weight.data shoud be taken from a normal distribution
            # m.weight.data.normal_(0.0,1/np.sqrt(y))
            nn.init.xavier_normal_(m.weight)
            # m.bias.data should be 0
            m.bias.data.fill_(0)
            print("Init linear")
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            # torch.nn.init.xavier_uniform(m.bias)
            print("Init conv")
        else:
            print(f"{type(m)} is not initialized")
    for seq in [attention_model.feature_extractor_part1, attention_model.feature_extractor_part2]:
        for layer in seq:
            weights_init_normal(layer)


def train_model(model, args: Args, train_loader,
                need_init_weights: bool = True,
                checkpoints_path: Optional[Path] = None):
    if need_init_weights:
        init_weights(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)

    def train_step(epoch):
        model.train()
        train_loss = 0.
        train_error = 0.
        for batch_idx, (data, label) in enumerate(train_loader):
            bag_label = label[0]
            if args.cuda:
                data, bag_label = data.cuda(), bag_label.cuda()
            data, bag_label = Variable(data).double(), Variable(bag_label).double()

            # reset gradients
            optimizer.zero_grad()
            # calculate loss and metrics
            loss, _ = model.calculate_objective(data, bag_label)
            
            if args.use_different_templates:
                # model.templates shape: (n_templates, template_size)
                template_penalty = model.templates @ model.templates.T
                template_penalty.fill_diagonal_(0.0)
                norms = torch.norm(model.templates, dim=1)
                normalization = norms[:, None] @ norms[None, :]
                normalization.fill_diagonal_(1.0)
                template_penalty /= normalization
                loss += 0.1 * torch.square(template_penalty).sum()
            
            train_loss += loss
            error, _ = model.calculate_classification_error(data, bag_label)
            train_error += error
            loss.backward()
            optimizer.step()

        # calculate loss and error for epoch
        train_loss /= len(train_loader)
        train_error /= len(train_loader)

        print('Epoch: {}, Loss: {:.4f}, Train error: {:.4f}'.format(epoch, train_loss.item(), train_error))
        return train_loss, train_error

    print('Start Training')
    for epoch in range(1, args.epochs + 1):
        train_loss, train_error = train_step(epoch)
        
        if checkpoints_path is not None:
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss,
                    'error': train_error,
                },
                str(checkpoints_path / f'{epoch}.pth')
            )
    return model
