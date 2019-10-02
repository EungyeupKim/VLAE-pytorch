from timeit import default_timer
from collections import defaultdict
from tqdm import trange
import torch
import numpy as np
from disvae.models.losses import _reconstruction_loss, _kl_normal_loss

from disvae.utils.modelIO import save_model


TRAIN_LOSSES_LOGFILE = "train_losses.log"


class Trainer():
    """
    Class to handle training of model.

    Parameters
    ----------
    model: disvae.vae.VAE

    optimizer: torch.optim.Optimizer

    loss_f: disvae.models.BaseLoss
        Loss function.

    device: torch.device, optional
        Device on which to run the code.

    logger: logging.Logger, optional
        Logger.

    save_dir : str, optional
        Directory for saving logs.

    gif_visualizer : viz.Visualizer, optional
        Gif Visualizer that should return samples at every epochs.

    is_progress_bar: bool, optional
        Whether to use a progress bar for training.
    """

    def __init__(self, model, optimizer, reg_coeff,
                 device=torch.device("cpu"),
                 save_dir="results",
                 gif_visualizer=None,
                 is_progress_bar=True):

        self.device = device
        self.model = model.to(self.device)
        self.reg_coeff = reg_coeff
        self.optimizer = optimizer
        self.save_dir = save_dir
        self.is_progress_bar = is_progress_bar
        self.gif_visualizer = gif_visualizer
        self.loss_ratio = 8.0

    def __call__(self, data_loader,
                 epochs=10,
                 checkpoint_every=10):
        """
        Trains the model.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader

        epochs: int, optional
            Number of epochs to train the model for.

        checkpoint_every: int, optional
            Save a checkpoint of the trained model every n epoch.
        """
        start = default_timer()
        self.model.train()
        for epoch in range(epochs):
            mean_epoch_loss = self._train_epoch(data_loader, epoch)
            print('Loss per epoch : {} ---- {}/{}'.format(mean_epoch_loss, epoch, epochs))

            if self.gif_visualizer is not None:
                self.gif_visualizer(epoch)

            if epoch % checkpoint_every == 0:
                save_model(self.model, self.save_dir, filename="model-{}.pt".format(epoch))

        if self.gif_visualizer is not None:
            self.gif_visualizer.save_reset()

        self.model.eval()

        delta_time = (default_timer() - start) / 60
        print('Finished training after {:.1f} min.'.format(delta_time))

    def _train_epoch(self, data_loader, epoch):
        """
        Trains the model for one epoch.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader

        storer: dict
            Dictionary in which to store important variables for vizualisation.

        epoch: int
            Epoch number

        Return
        ------
        mean_epoch_loss: float
            Mean loss per image
        """
        epoch_loss = 0.
        kwargs = dict(desc="Epoch {}".format(epoch + 1), leave=False,
                      disable=not self.is_progress_bar)
        with trange(len(data_loader), **kwargs) as t:
            for _, (data, _) in enumerate(data_loader):
                iter_loss = self._train_iteration(data)
                epoch_loss += iter_loss

                t.set_postfix(loss=iter_loss)
                t.update()

        mean_epoch_loss = epoch_loss / len(data_loader)
        return mean_epoch_loss

    def _train_iteration(self, data):
        """
        Trains the model for one iteration on a batch of data.

        Parameters
        ----------
        data: torch.Tensor
            A batch of data. Shape : (batch_size, channel, height, width).

        storer: dict
            Dictionary in which to store important variables for vizualisation.
        """
        data = data.to(self.device)

        gen_img, h0_ladd_mean, h0_ladd_stddev, h1_ladd_mean, h1_ladd_stddev, h2_ladd_mean, h2_ladd_stddev = self.model(data)
        recon_loss = torch.mean(torch.abs(data - gen_img))
        kl_normal_loss_0 = ((-1 * torch.log(h0_ladd_stddev) + 0.5 * (h0_ladd_stddev ** 2) + 0.5 * (h0_ladd_mean ** 2) - 0.5).mean(dim=0)).sum()
        kl_normal_loss_1 = ((-1 * torch.log(h1_ladd_stddev) + 0.5 * (h1_ladd_stddev ** 2) + 0.5 * (h1_ladd_mean ** 2) - 0.5).mean(dim=0)).sum()
        kl_normal_loss_2 = ((-1 * torch.log(h2_ladd_stddev) + 0.5 * (h2_ladd_stddev ** 2) + 0.5 * (h2_ladd_mean ** 2) - 0.5).mean(dim=0)).sum()

        kl_loss = self.reg_coeff[0] * kl_normal_loss_0 + \
                  self.reg_coeff[1] * kl_normal_loss_1 + \
                  self.reg_coeff[2] * kl_normal_loss_2

        loss = kl_loss + recon_loss * np.prod(self.model.img_size) * self.loss_ratio

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()