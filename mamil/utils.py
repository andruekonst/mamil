from pathlib import Path
import numpy as np
import torch
from torch.autograd import Variable
from matplotlib import pyplot as plt
from collections import Counter
from typing import Optional


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


def extract_patches(image: torch.Tensor, size: int, stride: Optional[int] = None) -> torch.Tensor:
    """Extracts square patches from the image.

    Args:
      image: Input image of shape `(height, width, n_channels)`.
      size: Patch size.
      stride: Stride. If None, `stride == size`.
    Returns:
      Tensor with patches of shape `(n_patches_h, n_patches_w, n_channels, size, size)`.
    """
    if stride is None:
        stride = size
    return image.unfold(0, size, stride).unfold(1, size, stride)


def make_test_plots(model,
                    plots_data_loader,
                    args,
                    n_images: int = 5,
                    only: Optional[str] = None,
                    verbose: bool = False,
                    save_dir: Optional[Path] = None):

    model.eval()
    c = 0
    for _batch_idx, (data, label) in enumerate(plots_data_loader):
        bag_label = label[0]
        instance_labels = label[1]
        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data).double(), Variable(bag_label).double()
        _loss, attention_weights = model.calculate_objective(data, bag_label)
        _error, predicted_label = model.calculate_classification_error(data, bag_label)

        if only == 'positive':
            if not bag_label:
                continue
        elif only == 'negative':
            if bag_label:
                continue
                
        bag_level = (bag_label.cpu().data.numpy()[0], int(predicted_label.cpu().data.numpy()[0][0]))
        gt_label = bool(bag_level[0])
        pred_label = bool(predicted_label.item())
        
        if only == 'true_positive':
            if not (gt_label and pred_label):
                continue
        elif only == 'false_negative':
            if not (gt_label and not pred_label):
                continue
        elif only == 'true_negative':
            if not (not gt_label and not pred_label):
                continue
        elif only == 'false_positive':
            if not (not gt_label and pred_label):
                continue

        c += 1
        if c <= n_images:
            instance_level = list(zip(instance_labels.numpy()[0].tolist(),
                                    np.round(attention_weights.cpu().data.numpy()[0], decimals=3).tolist()))

            if verbose:
                print('\nTrue Bag Label, Predicted Bag Label: {}\n'
                        'True Instance Labels, Attention Weights: {}'.format(bag_level, instance_level))
            il = instance_labels.numpy()[0].tolist()
            aw = attention_weights.cpu().data.numpy()[0]
            fig, ax = plt.subplots(1, len(il), figsize=(16, 2), dpi=150)
            fig.suptitle(f'Bag label: {gt_label}, predicted: {pred_label}')
            for i in range(len(il)):
                digit_im = data[0, i, 0].detach().cpu().numpy()
                digit_im_rgb = np.stack((digit_im,) * 3, axis=2)
                label_bar = np.ones((3, 28, 3)) * aw[i]
                label_bar *= digit_im.max() / aw.max()
                label_bar[..., [0, 2]] = 0
                digit_im = np.concatenate((digit_im_rgb, label_bar), axis=0)
                digit_im /= digit_im.max()
                np.clip(digit_im, 0, 1, out=digit_im)
                ax[i].imshow(digit_im)  # , cmap='gray')
                ax[i].set_axis_off()
                ax[i].set_title(f'{aw[i]:.3f}', y=-0.2)
            if save_dir is not None:
                save_dir.mkdir(parents=True, exist_ok=True)
                plt.savefig(str(save_dir / f'{c:02d}.png'), dpi=200)
