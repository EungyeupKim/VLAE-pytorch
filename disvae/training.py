from timeit import default_timer
from collections import defaultdict
from tqdm import trange
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

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

    def __call__(self, args, data_loader,
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
        self.args = args
        self.model.train()

        if self.args.exp:
            for epoch in range(epochs):
                if self.args.cl_vae:
                    mean_epoch_loss, mean_epoch_vlae_loss, mean_epoch_cl_loss = self._train_epoch(data_loader, epoch)
                    print('Train Set: Loss_epoch : {:.4f}, VLAE_epoch : {:.4f}, CL_epoch : {:.4f} ---- {}/{}'.format(mean_epoch_loss, mean_epoch_vlae_loss, mean_epoch_cl_loss, epoch, epochs))
                else:
                    mean_epoch_loss = self._train_epoch(data_loader, epoch)
                    print('Train Set: Loss_epoch : {:.4f} ---- {}/{}'.format(mean_epoch_loss, epoch, epochs))

                if not self.args.cl_loss:
                    if self.gif_visualizer is not None:
                        self.gif_visualizer(epoch, viz_single=self.args.viz_single)

                ###evaluation###
                # self.model.eval()
                # eval_acc = 0
                # eval_loss = 0
                # with torch.no_grad():
                #     for _, (data, label) in enumerate(test_data_loader):
                #         data = data.to(self.device)
                #         label = label.to(self.device)
                #         output = self.model(data)
                #         expect_class = output.argmax(dim=1, keepdim=True)
                #         eval_loss += F.nll_loss(output, label, reduction='sum').item()
                #         eval_acc += expect_class.eq(label.view_as(expect_class)).sum().item()
                #     eval_loss /= len(test_data_loader.dataset)
                #     eval_acc /= len(test_data_loader.dataset)
                #     print('Test Set: Loss_epoch: {}, Accuracy_epoch: {} ---- {}/{}'.format(eval_loss, eval_acc, epoch, epochs))
                ####

                if epoch % checkpoint_every == 0:
                    save_model(self.model, self.save_dir, filename="model-{}.pt".format(epoch))

            # if self.gif_visualizer is not None:
            #     self.gif_visualizer.save_reset()
        else:
            for epoch in range(epochs):
                mean_epoch_loss = self._train_epoch(data_loader, epoch)
                print('Loss per epoch : {} ---- {}/{}'.format(mean_epoch_loss, epoch, epochs))

                if self.gif_visualizer is not None:
                    self.gif_visualizer(epoch, viz_single=self.args.viz_single)

                if epoch % checkpoint_every == 0:
                    save_model(self.model, self.save_dir, filename="model-{}.pt".format(epoch))

            # if self.gif_visualizer is not None:
            #     self.gif_visualizer.save_reset()

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
        if self.args.exp:
            epoch_loss = 0.
            epoch_vlae_loss = 0.
            epoch_cl_loss = 0.
            kwargs = dict(desc="Epoch {}".format(epoch + 1), leave=False,
                          disable=not self.is_progress_bar)
            if self.args.cl_loss:
                with trange(len(data_loader), **kwargs) as t:
                    for _, (data, label) in enumerate(data_loader):
                        iter_loss = self._train_iteration(data, label)
                        epoch_loss += iter_loss

                        t.set_postfix(loss=iter_loss)
                        t.update()

                mean_epoch_loss = epoch_loss / len(data_loader.dataset)

                return mean_epoch_loss
            if self.args.cl_vae:
                with trange(len(data_loader), **kwargs) as t:
                    for _, (data, label) in enumerate(data_loader):
                        iter_loss, vlae_loss, cl_loss = self._train_iteration(data, label)
                        epoch_loss += iter_loss
                        epoch_vlae_loss += vlae_loss
                        epoch_cl_loss += cl_loss

                        t.set_postfix(loss=iter_loss)
                        t.update()

                mean_epoch_loss = epoch_loss / len(data_loader)
                mean_epoch_vlae_loss = epoch_vlae_loss / len(data_loader)
                mean_epoch_cl_loss = epoch_cl_loss / len(data_loader)

                return mean_epoch_loss, mean_epoch_vlae_loss, mean_epoch_cl_loss
            else:
                with trange(len(data_loader), **kwargs) as t:
                    for _, (data, label) in enumerate(data_loader):
                        iter_loss = self._train_iteration(data, label)
                        epoch_loss += iter_loss

                        t.set_postfix(loss=iter_loss)
                        t.update()

                mean_epoch_loss = epoch_loss / len(data_loader)
                return mean_epoch_loss
        else:
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



    def _train_iteration(self, data, label):
        """
        Trains the model for one iteration on a batch of data.

        Parameters
        ----------
        data: torch.Tensor
            A batch of data. Shape : (batch_size, channel, height, width).

        storer: dict
            Dictionary in which to store important variables for vizualisation.
        """
        if self.args.exp:
            if self.args.cl_loss:
                data = data.to(self.device)
                label = label.to(self.device)
                output = self.model(data)

                cl_loss = F.nll_loss(output, label) * 100

                loss = cl_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                return loss.item()
            elif self.args.cl_vae:

                data = data.to(self.device)
                label = label.to(self.device)

                output, gen_img, h0_ladd_mean, h0_ladd_stddev, h1_ladd_mean, h1_ladd_stddev, h2_ladd_mean, h2_ladd_stddev = self.model(data)

                cl_loss = F.nll_loss(output, label) * 100
                recon_loss = torch.mean(torch.abs(data - gen_img))
                kl_normal_loss_0 = ((-1 * torch.log(h0_ladd_stddev) + 0.5 * (h0_ladd_stddev ** 2) + 0.5 * (h0_ladd_mean ** 2) - 0.5).mean(dim=0)).sum()
                kl_normal_loss_1 = ((-1 * torch.log(h1_ladd_stddev) + 0.5 * (h1_ladd_stddev ** 2) + 0.5 * (h1_ladd_mean ** 2) - 0.5).mean(dim=0)).sum()
                kl_normal_loss_2 = ((-1 * torch.log(h2_ladd_stddev) + 0.5 * (h2_ladd_stddev ** 2) + 0.5 * (h2_ladd_mean ** 2) - 0.5).mean(dim=0)).sum()

                kl_loss = self.reg_coeff[0] * kl_normal_loss_0 + \
                          self.reg_coeff[1] * kl_normal_loss_1 + \
                          self.reg_coeff[2] * kl_normal_loss_2

                vlae_loss = kl_loss + recon_loss * np.prod(self.model.img_size) * self.loss_ratio
                loss = vlae_loss + cl_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                return loss.item(), vlae_loss, cl_loss
            else:
                data = data.to(self.device)

                gen_img, h0_ladd_mean, h0_ladd_stddev, h1_ladd_mean, h1_ladd_stddev, h2_ladd_mean, h2_ladd_stddev = self.model(data)

                recon_loss = torch.mean(torch.abs(data - gen_img))
                kl_normal_loss_0 = (
                    (-1 * torch.log(h0_ladd_stddev) + 0.5 * (h0_ladd_stddev ** 2) + 0.5 * (h0_ladd_mean ** 2) - 0.5).mean(
                        dim=0)).sum()
                kl_normal_loss_1 = (
                    (-1 * torch.log(h1_ladd_stddev) + 0.5 * (h1_ladd_stddev ** 2) + 0.5 * (h1_ladd_mean ** 2) - 0.5).mean(
                        dim=0)).sum()
                kl_normal_loss_2 = (
                    (-1 * torch.log(h2_ladd_stddev) + 0.5 * (h2_ladd_stddev ** 2) + 0.5 * (h2_ladd_mean ** 2) - 0.5).mean(
                        dim=0)).sum()

                kl_loss = self.reg_coeff[0] * kl_normal_loss_0 + \
                          self.reg_coeff[1] * kl_normal_loss_1 + \
                          self.reg_coeff[2] * kl_normal_loss_2

                loss = kl_loss + recon_loss * np.prod(self.model.img_size) * self.loss_ratio

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                return loss.item()
        else:
            data = data.to(self.device)

            gen_img, h0_ladd_mean, h0_ladd_stddev, h1_ladd_mean, h1_ladd_stddev, h2_ladd_mean, h2_ladd_stddev = self.model(
                data)
            recon_loss = torch.mean(torch.abs(data - gen_img))
            kl_normal_loss_0 = (
                (-1 * torch.log(h0_ladd_stddev) + 0.5 * (h0_ladd_stddev ** 2) + 0.5 * (h0_ladd_mean ** 2) - 0.5).mean(
                    dim=0)).sum()
            kl_normal_loss_1 = (
                (-1 * torch.log(h1_ladd_stddev) + 0.5 * (h1_ladd_stddev ** 2) + 0.5 * (h1_ladd_mean ** 2) - 0.5).mean(
                    dim=0)).sum()
            kl_normal_loss_2 = (
                (-1 * torch.log(h2_ladd_stddev) + 0.5 * (h2_ladd_stddev ** 2) + 0.5 * (h2_ladd_mean ** 2) - 0.5).mean(
                    dim=0)).sum()

            kl_loss = self.reg_coeff[0] * kl_normal_loss_0 + \
                      self.reg_coeff[1] * kl_normal_loss_1 + \
                      self.reg_coeff[2] * kl_normal_loss_2

            loss = kl_loss + recon_loss * np.prod(self.model.img_size) * self.loss_ratio

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            return loss.item()