from collections import OrderedDict
import torch
import logging
import numpy as np
from mmskeleton.utils import call_obj, import_obj #, load_checkpoint
from mmcv.runner import Runner
from mmcv import Config, ProgressBar
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from sklearn.metrics import confusion_matrix
import scikitplot as skplt
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def test(model_cfg, dataset_cfg, checkpoint, batch_size=64, gpus=1, workers=0):
    dataset = call_obj(**dataset_cfg)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=workers)

    # put model on gpus
    if isinstance(model_cfg, list):
        model = [call_obj(**c) for c in model_cfg]
        model = torch.nn.Sequential(*model)
    else:
        model = call_obj(**model_cfg)
    load_checkpoint(model, checkpoint, map_location='cpu')
    model = MMDataParallel(model, device_ids=range(gpus)).cuda()
    model.eval()

    results = []
    labels = []
    prog_bar = ProgressBar(len(dataset))
    for data, label in data_loader:
        with torch.no_grad():
            output = model(data).data.cpu().numpy()
        results.append(output)
        labels.append(label)
        for i in range(len(data)):
            prog_bar.update()
    results = np.concatenate(results)
    labels = np.concatenate(labels)

    print('Top 1: {:.2f}%'.format(100 * topk_accuracy(results, labels, 1)))
    #print('Top 5: {:.2f}%'.format(100 * topk_accuracy(results, labels, 5)))





def train(
        work_dir,
        model_cfg,
        loss_cfg,
        dataset_cfg,
        optimizer_cfg,
        batch_size,
        total_epochs,
        training_hooks,
        workflow=[('train', 1)],
        gpus=1,
        log_level=0,
        workers=4,
        resume_from=None,
        load_from=None,
):

    # prepare data loaders
    if isinstance(dataset_cfg, dict):
        dataset_cfg = [dataset_cfg]
    data_loaders = [
        torch.utils.data.DataLoader(dataset=call_obj(**d),
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=workers,
                                    drop_last=True) for d in dataset_cfg
    ]

    # put model on gpus
    if isinstance(model_cfg, list):
        model = [call_obj(**c) for c in model_cfg]
        model = torch.nn.Sequential(*model)
    else:
        model = call_obj(**model_cfg)
    model.apply(weights_init)
    model = MMDataParallel(model, device_ids=range(gpus)).cuda()
    loss = call_obj(**loss_cfg)

    # build runner
    optimizer = call_obj(params=model.parameters(), **optimizer_cfg)
    runner = Runner(model, batch_processor, optimizer, work_dir, log_level)
    runner.register_training_hooks(**training_hooks)

    if resume_from:
        runner.resume(resume_from)
    elif load_from:
        runner.load_checkpoint(load_from)

    # run
    workflow = [tuple(w) for w in workflow]
    runner.run(data_loaders, workflow, total_epochs, loss=loss)


# process a batch of data
def batch_processor(model, datas, train_mode, loss):

    data, label = datas
    data = data.cuda()
    label = label.cuda()

    # forward
    output = model(data)
    losses = loss(output, label)

    # output
    log_vars = dict(loss=losses.item())
    if not train_mode:
        log_vars['top1'] = topk_accuracy(output, label)
        log_vars['top5'] = topk_accuracy(output, label, 5)

    outputs = dict(loss=losses, log_vars=log_vars, num_samples=len(data.data))
    return outputs


def topk_accuracy(score, label, k=1):
    rank = score.argsort()
    hit_top_k = [l in rank[i, -k:] for i, l in enumerate(label)]
    accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
    predictions = [rank[i][-1] for i in range(len(rank))]
    #classes = ['ADL', 'FALL']
    classes = ['1', '2', '3', '4', '5', '6', '7', '8']

    """""
    # print confusion matrix
    plt.figure()
    skplt.metrics.plot_confusion_matrix(label, predictions, normalize=False,
                                        title='Confusion matrix, without normalization')
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.show()
    """""

    cnf_mat = confusion_matrix(label, predictions)
    print_confusion_matrix(cnf_mat,classes)
    plt.show()


    return accuracy


def print_confusion_matrix(confusion_matrix, class_names, figsize=(10, 7), fontsize=14):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.

    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix.
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.

    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig


def weights_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv1d') != -1:
        model.weight.data.normal_(0.0, 0.02)
        if model.bias is not None:
            model.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        model.weight.data.normal_(0.0, 0.02)
        if model.bias is not None:
            model.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)