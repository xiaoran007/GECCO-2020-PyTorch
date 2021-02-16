"""Module containing the Generator class."""
import torch


class Generator(torch.nn.Module):
    """Class that defines the Generator."""

    def __init__(self,
                 size_latent: int,
                 size_labels: int,
                 num_hidden_layers: int,
                 size_outputs: int,
                 **kwargs):
        """
        Function to initialize the object.

        Parameters
        ----------
        latent_size: int
        num_labels: int
        num_hidden_layers: int
        size_outputs: int

        """
        super(Generator, self).__init__()

        assert isinstance(size_latent, int) and (size_latent > 0)
        assert isinstance(size_labels, int) and (size_labels > 0)
        assert isinstance(num_hidden_layers, int) and (num_hidden_layers > 0)
        assert isinstance(size_outputs, int) and (size_outputs > 0)

        list_num_nodes: list = torch.linspace(start=256,
                                              end=int(256 * num_hidden_layers),
                                              steps=num_hidden_layers,
                                              requires_grad=False).int().tolist()

        self._embedding: torch.nn.Embedding = torch.nn.Embedding(size_labels, size_labels)

        self._nn: torch.nn.Sequential = torch.nn.Sequential()
        idx: int = 1
        for (i, o) in zip([size_latent + size_labels] + list_num_nodes[:-1], list_num_nodes):
            self._nn.add_module(name="linear_{0}".format(idx),
                                module=torch.nn.Linear(in_features=i, out_features=o, bias=True))
            self._nn.add_module(name="leakyrelu_{0}".format(idx),
                                module=torch.nn.LeakyReLU(negative_slope=0.2, inplace=True))
            self._nn.add_module(name="batchnorm_{0}".format(idx),
                                module=torch.nn.BatchNorm1d(num_features=o, momentum=0.8))
            idx = idx + 1
        self._nn.add_module(name="linear_{0}".format(idx),
                            module=torch.nn.Linear(in_features=list_num_nodes[-1],
                                                   out_features=size_outputs,
                                                   bias=True))
        self._nn.add_module(name="tanh", module=torch.nn.Tanh())

    def forward(self, latent_vector: torch.Tensor, labels: torch.Tensor, **kwargs):
        """Function to perform computation."""
        x: torch.Tensor = torch.cat((self._embedding(labels), latent_vector), dim=-1)

        return self._nn(x)
