"""Module containing the functions."""
import copy
import os
import time
from typing import Tuple

import numpy as np
import torch


def train(generator: torch.nn.Module,
          discriminator: torch.nn.Module,
          x: torch.Tensor,
          y: torch.Tensor,
          latent_size: int,
          batch_size: int = 16,
          num_epochs: int = 2,
          run_device: str = "cpu",
          learning_rate: float = 0.001,
          beta_1: float = 0.9,
          beta_2: float = 0.999,
          random_state: torch.ByteTensor = torch.get_rng_state().clone(),
          verbose: bool = False) -> Tuple[torch.nn.Module, torch.nn.Module, torch.ByteTensor]:
    """
    Function to train GAN.

    Parameters
    ----------
    generator: torch.nn.Module
    discriminator: torch.nn.Module
    x: torch.Tensor
    y: torch.Tensor
    latent_size: int
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
    Tuple[torch.nn.Module, torch.nn.Module, torch.Tensor]

    """
    assert isinstance(generator, torch.nn.Module)
    assert isinstance(discriminator, torch.nn.Module)
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert isinstance(latent_size, int) and (latent_size > 0)
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

    # Set the generator and discriminator.
    net_g: torch.nn.Module = copy.deepcopy(generator.cpu())
    net_d: torch.nn.Module = copy.deepcopy(discriminator.cpu())
    run_device_train = run_device.lower()
    if run_device_train == "cuda":
        assert torch.cuda.is_available()
        net_g, net_d = net_g.cuda(), net_d.cuda()
        if torch.cuda.device_count() > 1:
            num_gpus: int = torch.cuda.device_count()
            net_g = torch.nn.DataParallel(net_g, device_ids=list(range(0, num_gpus)))
            net_d = torch.nn.DataParallel(net_d, device_ids=list(range(0, num_gpus)))

    # Set a criterion and optimizer.
    # criterion = torch.nn.MSELoss()
    criterion = torch.nn.BCELoss()
    optimizer_g = torch.optim.Adam(params=net_g.parameters(),
                                   lr=learning_rate,
                                   betas=(beta_1, beta_2))
    optimizer_d = torch.optim.Adam(params=net_d.parameters(),
                                   lr=learning_rate,
                                   betas=(beta_1, beta_2))

    # Covert PyTorch's Tensor to TensorDataset.
    x_train, y_train = x.clone(), y.clone()
    size_labels: int = int(y_train.max().item() - y_train.min().item()) + 1
    dataset = torch.utils.data.TensorDataset(x_train, y_train)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=0,
                                             shuffle=True)

    log_template: str = "[{0}/{1}] Time: {2:.2f}s"

    # Train the models.
    net_g.train()
    net_d.train()
    for epoch in range(1, num_epochs + 1):
        i = 0
        start_time: float = time.time()
        for (_, batch) in enumerate(dataloader, 0):
            batch_x, batch_y = batch
            batch_size: int = batch_x.size(0)

            real: torch.Tensor = torch.full(size=(batch_size,),
                                            fill_value=1,
                                            dtype=torch.float32,
                                            requires_grad=False)
            fake: torch.Tensor = torch.full(size=(batch_size,),
                                            fill_value=0,
                                            dtype=torch.float32,
                                            requires_grad=False)

            latent_vector: torch.Tensor = torch.randn(size=(batch_size, latent_size),
                                                      dtype=torch.float32)
            fake_y: torch.Tensor = torch.randint(low=0, high=size_labels, size=(batch_size,), dtype=torch.long)

            if run_device_train == "cuda":
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
                real, fake = real.cuda(), fake.cuda()
                latent_vector = latent_vector.cuda()
                fake_y = fake_y.cuda()

            fake_x: torch.Tensor = net_g(latent_vector, fake_y)

            # Discriminator
            optimizer_d.zero_grad()
            output_1: torch.Tensor = net_d(batch_x, batch_y)
            output_1 = output_1.view(size=(-1,))
            loss_d_real: torch.Tensor = criterion(output_1, real)
            output_2: torch.Tensor = net_d(fake_x, fake_y)
            output_2 = output_2.view(size=(-1,))
            loss_d_fake: torch.Tensor = criterion(output_2, fake)
            loss_d: torch.Tensor = loss_d_real + loss_d_fake
            loss_d.backward(retain_graph=True)
            optimizer_d.step()

            # Generator
            output_3: torch.Tensor = net_d(fake_x, fake_y)
            output_3 = output_3.view(size=(-1,))
            loss_g: torch.Tensor = criterion(output_3, real)
            optimizer_g.zero_grad()
            loss_g.backward(retain_graph=True)
            optimizer_g.step()

            # Print
            print(f"epoch: {epoch}, inner: {i} loss_g: {loss_g.item()}, loss_d: {loss_d.item()}")
            i += 1
        end_time: float = time.time()

        if verbose:
            print(log_template.format(epoch, num_epochs, end_time - start_time))

    if isinstance(net_g, torch.nn.DataParallel):
        net_g = net_g.module
    if isinstance(net_d, torch.nn.DataParallel):
        net_d = net_d.module

    random_state_after: torch.ByteTensor = torch.get_rng_state().clone()
    torch.set_rng_state(random_state_previous)

    return net_g.cpu(), net_d.cpu(), random_state_after.clone()


def predict(generator: torch.nn.Module,
            discriminator: torch.nn.Module,
            latent_size: int,
            output_by_label: dict,
            run_device: str = "cpu",
            random_state: torch.ByteTensor = torch.get_rng_state().clone(),
            verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function to generate sa
    Parameters
    ----------
    generator: torch.nn.Module
    discriminator: torch.nn.Module
    latent_size: int
    output_by_label: dict
    run_device: str
    random_state: torch.ByteTensor
    verbose: bool

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]

    """
    assert isinstance(generator, torch.nn.Module)
    assert isinstance(discriminator, torch.nn.Module)
    assert isinstance(latent_size, int) and (latent_size > 0)
    assert isinstance(output_by_label, dict)
    assert isinstance(run_device, str) and (run_device.lower() in ["cpu", "cuda"])
    assert isinstance(random_state, torch.ByteTensor)
    assert isinstance(verbose, bool)

    # Set the seed for generating random numbers.
    random_state_previous: torch.ByteTensor = torch.get_rng_state().clone()
    torch.set_rng_state(random_state)

    # Set the generator and discriminator.
    net_g: torch.nn.Module = copy.deepcopy(generator.cpu())
    net_d: torch.nn.Module = copy.deepcopy(discriminator.cpu())
    run_device_predict = run_device.lower()
    if run_device_predict == "cuda":
        assert torch.cuda.is_available()
        net_g, net_d = net_g.cuda(), net_d.cuda()
        if torch.cuda.device_count() > 1:
            num_gpus: int = torch.cuda.device_count()
            net_g = torch.nn.DataParallel(net_g, device_ids=list(range(0, num_gpus)))
            net_d = torch.nn.DataParallel(net_d, device_ids=list(range(0, num_gpus)))

    keys: list = list(output_by_label.keys())
    labels: torch.Tensor = torch.as_tensor([key for key in keys for _ in range(output_by_label[key])],
                                           dtype=torch.long)
    latent_vector: torch.Tensor = torch.randn(size=(len(labels), latent_size),
                                              dtype=torch.float32,
                                              requires_grad=False)
    if run_device_predict == "cuda":
        labels = labels.cuda()
        latent_vector = latent_vector.cuda()

    net_g.eval()
    net_d.eval()
    with torch.no_grad():
        output: torch.Tensor = net_g(latent_vector, labels).detach().cpu()

    labels = labels.cpu()

    torch.set_rng_state(random_state_previous)

    return output.numpy(), labels.numpy()


def save_model(generator: torch.nn.Module,
               discriminator: torch.nn.Module,
               model_path: str,
               random_state: torch.ByteTensor = torch.get_rng_state().clone()) -> bool:
    """
    Function to save the parameters of the trained generator and discriminator.

    Parameters
    ----------
    generator: torch.nn.Module
    discriminator: torch.nn.Module
    model_path: str
    random_state: torch.ByteTensor

    Returns
    -------
    bool

    """
    assert isinstance(generator, torch.nn.Module)
    assert isinstance(discriminator, torch.nn.Module)
    assert isinstance(model_path, str)
    assert isinstance(random_state, torch.ByteTensor)

    model_path = os.path.abspath(model_path)
    model_pardir: str = os.path.split(model_path)[0]
    if not os.path.exists(model_pardir):
        os.makedirs(model_pardir)

    checkpoint: dict = {
        "generator_state_dict": generator.cpu().state_dict().copy(),
        "discriminator_state_dict": discriminator.cpu().state_dict().copy(),
        "random_state": random_state.clone()
    }

    # Save the trained models.
    torch.save(checkpoint, model_path)

    return True


def load_model(generator: torch.nn.Module,
               discriminator: torch.nn.Module,
               model_path: str) -> Tuple[torch.nn.Module, torch.nn.Module, torch.ByteTensor]:
    """
    Function to load the parameters from the trained classifier.

    Parameters
    ----------
    generator: torch.nn.Module
    discriminator: torch.nn.Module
    model_path: str

    Returns
    -------
    Tuple[torch.nn.Module, torch.nn.Module, torch.ByteTensor]

    """
    assert isinstance(generator, torch.nn.Module)
    assert isinstance(discriminator, torch.nn.Module)
    assert os.path.splitext(model_path)[1].lower() in [".pt", ".pth"]

    checkpoint: dict = torch.load(model_path, map_location=torch.device("cpu"))
    if "generator_state_dict" not in checkpoint:
        raise KeyError("generator_state_dict")
    if "discriminator_state_dict" not in checkpoint:
        raise KeyError("discriminator_state_dict")

    # Load the trained generator.
    net_g = copy.deepcopy(generator.cpu())
    net_g.load_state_dict(checkpoint["generator_state_dict"])

    # Load the trained discriminator.
    net_d = copy.deepcopy(discriminator.cpu())
    net_d.load_state_dict(checkpoint["discriminator_state_dict"])

    # Load the random state
    random_state_load: torch.ByteTensor = checkpoint.get("random_state", torch.get_rng_state().clone())

    return net_g.cpu(), net_d.cpu(), random_state_load.clone()
