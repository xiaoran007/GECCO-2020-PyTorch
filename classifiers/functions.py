"""Module containing the functions."""
import copy
import os
import time
from collections import OrderedDict
from pprint import pprint
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn import metrics as sklearn_metrics


class _EarlyStopping:
    """Early stops the training, if validation loss doesn't improve after a given patience."""

    def __init__(self, patience: int = 7, delta: float = 0.0, verbose: bool = False):
        """
        Parameters
        ----------
        patience: int
            How long to wait after last time validation loss improved.
            Default: 7
        delta: float
            Minimum change in the monitored quantity to qualify as an improvement.
            Default: 0.0
        verbose: bool
            If True, prints a message for each validation loss improvement.
            Default: False

        """
        self.patience: int = patience
        self.delta: float = delta
        self.verbose: bool = verbose

        self.counter: int = 0
        self.early_stop: bool = False
        self.loss_min: float = np.Inf

        self.best_score: float = 0.0
        self.best_model: OrderedDict = OrderedDict()
        self.random_state: torch.ByteTensor = torch.ByteTensor()

    def __call__(self, loss, model):
        score: float = -loss

        if self.best_score == 0.0:
            self.best_score = score
            self.best_model = model.state_dict().copy()
            self.random_state = torch.get_rng_state().clone()
        elif score < (self.best_score + self.delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

            if self.verbose:
                print("EarlyStopping counter: {0} out of {1}".format(self.counter, self.patience))
        else:
            self.best_score = score
            self.best_model = model.state_dict().copy()
            self.random_state = torch.get_rng_state().clone()
            self.counter = 0

    def get_best_model(self):
        return self.best_model.copy(), self.random_state.clone()


def train(classifier: torch.nn.Module,
          x: torch.Tensor,
          y: torch.Tensor,
          test_x: torch.Tensor = torch.Tensor(),
          test_y: torch.Tensor = torch.Tensor(),
          batch_size: int = 16,
          num_epochs: int = 2,
          run_device: str = "cpu",
          learning_rate: float = 0.001,
          beta_1: float = 0.9,
          beta_2: float = 0.999,
          random_state: torch.ByteTensor = torch.get_rng_state().clone(),
          verbose: bool = False) -> Tuple[torch.nn.Module, torch.ByteTensor]:
    """
    Function to train classifiers and save the trained classifiers.

    Parameters
    ----------
    classifier: torch.nn.Module
    x: torch.Tensor
    y: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor
    batch_size: int
    num_epochs: int
    run_device: str
    learning_rate: float
    beta_1: float
    beta_2: float
    random_state: torch.ByteTensor
    verbose: bool

    Returns
    -------
    Tuple[torch.nn.Module, torch.Tensor]

    """
    assert isinstance(classifier, torch.nn.Module)
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert isinstance(test_x, torch.Tensor)
    assert isinstance(test_y, torch.Tensor)
    assert isinstance(batch_size, int) and (batch_size > 0)
    assert isinstance(num_epochs, int) and (num_epochs > 0)
    assert isinstance(run_device, str) and (run_device.lower() in ["cpu", "cuda"])
    assert isinstance(learning_rate, float) and (learning_rate > 0.0)
    assert isinstance(beta_1, float) and (0.0 <= beta_1 < 1.0)
    assert isinstance(beta_2, float) and (0.0 <= beta_2 < 1.0)
    assert isinstance(random_state, torch.ByteTensor)
    assert isinstance(verbose, bool)

    # Set the seed for generating random numbers.
    random_state_previous: torch.ByteTensor = torch.get_rng_state().clone()
    torch.set_rng_state(random_state)

    # Set the classifier.
    classifier_train: torch.nn.Module = copy.deepcopy(classifier.cpu())
    run_device_train: str = run_device.lower()
    if run_device_train == "cuda":
        assert torch.cuda.is_available()
        classifier_train = classifier_train.cuda()
        if torch.cuda.device_count() > 1:
            num_gpus: int = torch.cuda.device_count()
            classifier_train = torch.nn.DataParallel(classifier_train, device_ids=list(range(0, num_gpus)))

    # Set a criterion and optimizer.
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=classifier_train.parameters(),
                                 lr=learning_rate,
                                 betas=(beta_1, beta_2))

    # Covert PyTorch's Tensor to TensorDataset.
    x_train, y_train = x.clone(), y.clone()
    dataset_train = torch.utils.data.TensorDataset(x_train, y_train)
    dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                                   batch_size=batch_size,
                                                   num_workers=0,
                                                   shuffle=True)

    has_test: bool = False
    if (test_x.size(0) > 0) and (test_y.size(0) > 0):
        x_test, y_test = test_x.clone(), test_y.clone()
        dataset_test = torch.utils.data.TensorDataset(x_test, y_test)
        dataloader_test = torch.utils.data.DataLoader(dataset_test,
                                                      batch_size=batch_size,
                                                      num_workers=0,
                                                      shuffle=False)
        has_test = True

    # Initialize the early_stopping object.
    early_stopping: _EarlyStopping = _EarlyStopping(patience=10, delta=0.0, verbose=False)

    log_template: str = "[{0}/{1}] Loss: {2:.4f}, Time: {3:.2f}s"
    log_template_test: str = "[{0}/{1}] Loss (Train): {2:.4f}, Loss (Test): {3:.4f}, Time: {4:.2f}s"

    list_loss: list = list()
    list_loss_test: list = list()

    # Train the classifiers.
    classifier_train.train()
    for epoch in range(1, num_epochs + 1):
        start_time: float = time.time()
        for (_, batch) in enumerate(dataloader_train, 0):
            batch_x, batch_y = batch
            if run_device_train == "cuda":
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

            optimizer.zero_grad()
            output: torch.Tensor = classifier_train(batch_x)
            loss: torch.Tensor = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

            list_loss.append(loss.detach().cpu().item())
        end_time: float = time.time()

        if has_test:
            classifier_train.eval()
            for (_, batch) in enumerate(dataloader_test, 0):
                batch_x, batch_y = batch
                if run_device_train == "cuda":
                    batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

                output = classifier_train(batch_x)
                loss = criterion(output, batch_y)

                list_loss_test.append(loss.detach().cpu().item())
            classifier_train.train()

            early_stopping(loss=np.mean(list_loss_test), model=classifier_train)
        else:
            early_stopping(loss=np.mean(list_loss), model=classifier_train)

        if verbose:
            if has_test:
                print(log_template_test.format(epoch,
                                               num_epochs,
                                               np.mean(list_loss),
                                               np.mean(list_loss_test),
                                               end_time - start_time))
            else:
                print(log_template.format(epoch,
                                          num_epochs,
                                          np.mean(list_loss),
                                          end_time - start_time))

        if early_stopping.early_stop:
            state_dict, rng_state = early_stopping.get_best_model()
            classifier_train.load_state_dict(state_dict)
            torch.set_rng_state(rng_state)
            break

    if isinstance(classifier_train, torch.nn.DataParallel):
        classifier_train = classifier_train.module

    random_state_after: torch.ByteTensor = torch.get_rng_state().clone()
    torch.set_rng_state(random_state_previous)

    return classifier_train.cpu(), random_state_after.clone()


def predict(classifier: torch.nn.Module,
            x: torch.Tensor,
            run_device: str = "cpu",
            random_state: torch.ByteTensor = torch.get_rng_state().clone()) -> np.ndarray:
    """
    Function to evaluate the trained classifiers.

    Parameters
    ----------
    classifier: torch.nn.Module
    x: torch.Tensor
    run_device: str
    random_state: torch.ByteTensor

    Returns
    -------
    numpy.ndarray

    """
    assert isinstance(classifier, torch.nn.Module)
    assert isinstance(x, torch.Tensor)
    assert isinstance(run_device, str) and (run_device.lower() in ["cpu", "cuda"])
    assert isinstance(random_state, torch.ByteTensor)

    # Set the seed for generating random numbers.
    random_state_previous: torch.ByteTensor = torch.get_rng_state().clone()
    torch.set_rng_state(random_state)

    # Set the classifiers.
    classifier_predict: torch.nn.Module = copy.deepcopy(classifier.cpu())
    run_device_predict: str = run_device.lower()
    if run_device_predict == "cuda":
        assert torch.cuda.is_available()
        classifier_predict = classifier_predict.cuda()
        if torch.cuda.device_count() > 1:
            num_gpus: int = torch.cuda.device_count()
            classifier_predict = torch.nn.DataParallel(classifier_predict, device_ids=list(range(0, num_gpus)))

    # Covert PyTorch's Tensor to TensorDataset.
    x_predict: torch.Tensor = x.clone()
    dataset = torch.utils.data.TensorDataset(x_predict)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=x_predict.size(0),
                                             num_workers=0,
                                             shuffle=False)

    batch_x: torch.Tensor = next(iter(dataloader))[0]
    if run_device_predict == "cuda":
        batch_x = batch_x.cuda()

    classifier_predict.eval()
    with torch.no_grad():
        output: torch.Tensor = classifier_predict(batch_x)
        predict_y: torch.Tensor = output.detach().cpu().argmax(dim=1, keepdim=True)

    torch.set_rng_state(random_state_previous)
    print(predict_y.numpy())
    return predict_y.numpy()


def evaluate(classifier: torch.nn.Module,
             x: torch.Tensor,
             y: torch.Tensor,
             metric: str = "f1_score",
             run_device: str = "cpu",
             random_state: torch.ByteTensor = torch.get_rng_state().clone(),
             verbose: bool = False) -> np.ndarray:
    """
    Function to evaluate the trained classifiers.

    Parameters
    ----------
    classifier: torch.nn.Module
    x: torch.Tensor
    y: torch.Tensor
    metric: str
    run_device: str
    random_state: torch.ByteTensor
    verbose: bool

    Returns
    -------
    numpy.ndarray

    """
    assert isinstance(classifier, torch.nn.Module)
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert isinstance(metric, str) and (metric.lower() in ["confusion_matrix", "f1_score"])
    assert isinstance(run_device, str) and (run_device.lower() in ["cpu", "cuda"])
    assert isinstance(random_state, torch.ByteTensor)
    assert isinstance(verbose, bool)

    predict_y: np.ndarray = predict(classifier=classifier,
                                    x=x,
                                    run_device=run_device,
                                    random_state=random_state)

    num_labels: int = int(y.max().item() - y.min().item()) + 1
    metric_evaluate = metric.lower()
    if metric_evaluate == "confusion_matrix":
        confusion_matrix: np.ndarray = sklearn_metrics.confusion_matrix(y.numpy(),
                                                                        predict_y,
                                                                        labels=list(range(0, num_labels)))
        if verbose:
            df_cm: pd.DataFrame = pd.DataFrame(confusion_matrix)
            df_cm.columns = ["Predict_{0}".format(label) for label in range(0, num_labels)]
            df_cm.index = ["Real_{0}".format(label) for label in range(0, num_labels)]

            print(">> Confusion matrix :")
            pprint(df_cm)

        return confusion_matrix
    elif metric_evaluate == "f1_score":
        f1_score: np.ndarray = sklearn_metrics.f1_score(y.numpy(),
                                                        predict_y,
                                                        labels=list(range(0, num_labels)),
                                                        average=None)

        if verbose:
            df_f1: pd.DataFrame = pd.DataFrame(f1_score[np.newaxis])
            df_f1.columns = ["Label_{0}".format(label) for label in range(0, num_labels)]
            df_f1.index = ["F1_score"]

            print(">> F1 score :")
            pprint(df_f1)

        return f1_score
    else:
        raise ValueError()


def save_model(classifier: torch.nn.Module,
               model_path: str,
               random_state: torch.ByteTensor = torch.get_rng_state().clone()) -> bool:
    """
    Function to save the parameters of the trained classifier.

    Parameters
    ----------
    classifier: torch.nn.Module
    model_path: str
    random_state: torch.ByteTensor

    Returns
    -------
    bool

    """
    assert isinstance(classifier, torch.nn.Module)
    assert isinstance(model_path, str)
    assert os.path.splitext(model_path)[1].lower() in [".pt", ".pth"]
    assert isinstance(random_state, torch.ByteTensor)

    model_path = os.path.abspath(model_path)
    model_pardir: str = os.path.split(model_path)[0]
    if not os.path.exists(model_pardir):
        os.makedirs(model_pardir)

    checkpoint: dict = {
        "classifier_state_dict": classifier.cpu().state_dict().copy(),
        "random_state": random_state.clone()
    }

    # Save the trained classifiers.
    torch.save(checkpoint, model_path)

    return True


def load_model(classifier: torch.nn.Module, model_path: str) -> Tuple[torch.nn.Module, torch.ByteTensor]:
    """
    Function to load the parameters from the trained classifier.

    Parameters
    ----------
    classifier: torch.nn.Module
    model_path: str

    Returns
    -------
    Tuple[torch.nn.Module, torch.ByteTensor]

    """
    assert isinstance(classifier, torch.nn.Module)
    assert isinstance(model_path, str) and os.path.exists(model_path)
    assert os.path.splitext(model_path)[1].lower() in [".pt", ".pth"]

    checkpoint: dict = torch.load(model_path, map_location=torch.device("cpu"))
    if "classifier_state_dict" not in checkpoint:
        raise KeyError("classifier_state_dict")

    # Load the trained classifiers.
    classifier_load: torch.nn.Module = copy.deepcopy(classifier.cpu())
    classifier_load.load_state_dict(checkpoint["classifier_state_dict"])

    # Load the random state
    random_state_load: torch.ByteTensor = checkpoint.get("random_state", torch.get_rng_state().clone())

    return classifier_load.cpu(), random_state_load.clone()
