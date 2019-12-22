import numpy as np
import torch
import torch.optim as optim
from utils import strokes2rgb
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer():
    def __init__(self, model, data_loader, tb_writer, checkpoint_dir=None, learning_rate=0.0001, wkl=1.0, eta_min=0.0, R=0.99999, clip_val=1.0):
        self.model = model
        self.data_loader = data_loader
        self.tb_writer = tb_writer
        self.enc_opt = optim.Adam(
            self.model.encoder.parameters(), lr=learning_rate)
        self.dec_opt = optim.Adam(
            self.model.decoder.parameters(), lr=learning_rate)
        self.checkpoint_dir = checkpoint_dir
        self.wkl = wkl
        self.clip_val = clip_val
        self.epoch = 0
        self.mininum_loss = -0.2  # from this loss, the trainer save models
        self.eta_min = eta_min
        self.R = R
        self.step_per_epoch = len(
            self.data_loader.dataset) / self.data_loader.batch_size
        # TODO plot Decoder Graph
        inputs = (self.data_loader.dataset[0])[0].unsqueeze(1)
        self.tb_writer.add_graph(self.model.encoder, inputs)
        #z, _, _ = self.model.encoder(inputs)
        #self.tb_writer.add_graph(self.model.decoder, (inputs, z))

    def train(self, epoch):
        for e in range(epoch):
            x = None
            step_in_epoch = 0
            for x, _ in tqdm(self.data_loader, ascii=True):
                x = x.permute(1, 0, 2)
                self.train_on_batch(x, step_in_epoch)
                step_in_epoch += 1
            self.epoch += 1

            # TODO fix proper batch to calculate loss
            with torch.no_grad():
                loss = self.loss_on_batch(x)
                self.tb_writer.add_scalar("loss/train", loss[0], self.epoch)
                self.tb_writer.add_scalar("loss/train/Ls", loss[1], self.epoch)
                self.tb_writer.add_scalar("loss/train/Lp", loss[2], self.epoch)
                self.tb_writer.add_scalar("loss/train/Lr", loss[3], self.epoch)
                self.tb_writer.add_scalar(
                    "loss/train/Lkl", loss[4], self.epoch)
                self.tb_writer.add_scalar(
                    "loss/train/wkl*eta*Lkl", loss[5], self.epoch)

                # Save model
                if self.epoch % 10 == 0:
                    if self.mininum_loss > loss[0]:
                        self.mininum_loss = loss[0]
                        torch.save(self.model.encoder.cpu(), str(
                            self.checkpoint_dir) + '/encoder-' + str(float(self.mininum_loss)) + '.pth')
                        torch.save(self.model.decoder.cpu(), str(
                            self.checkpoint_dir) + '/decoder-' + str(float(self.mininum_loss)) + '.pth')
                        # TODO save optimizer of cpu
                        torch.save(self.enc_opt, str(self.checkpoint_dir) +
                                   '/enc_opt-' + str(float(self.mininum_loss)) + '.pth')
                        torch.save(self.dec_opt, str(self.checkpoint_dir) +
                                   '/dec_opt-' + str(float(self.mininum_loss)) + '.pth')
                        self.model.encoder.to(device)
                        self.model.decoder.to(device)

            x = x[:, 0, :].unsqueeze(1)
            origial = x
            self.tb_writer.add_text(
                'reconstruction/original', str(origial), self.epoch)
            self.tb_writer.add_image(
                "reconstruction/original", strokes2rgb(origial), self.epoch)

            recon = self.model.reconstruct(x)
            self.tb_writer.add_text(
                'reconstruction/prediction', str(recon), self.epoch)
            self.tb_writer.add_image(
                "reconstruction/prediction", strokes2rgb(recon), self.epoch)

            self.tb_writer.flush()

    def train_on_batch(self, x, step_in_epoch=0):
        self.model.encoder.train()
        self.model.encoder.zero_grad()
        self.model.decoder.train()
        self.model.decoder.zero_grad()

        loss, _, _, _, _, _ = self.loss_on_batch(x, step_in_epoch)
        loss.backward()

        torch.nn.utils.clip_grad_value_(
            self.model.encoder.parameters(), self.clip_val)
        torch.nn.utils.clip_grad_value_(
            self.model.decoder.parameters(), self.clip_val)

        self.enc_opt.step()
        self.dec_opt.step()

    def loss_on_batch(self, x, step_in_epoch=0):
        batch_size = x.shape[1]

        z, mu, sigma_hat = self.model.encoder(x)

        sos = torch.stack(
            [torch.tensor([0, 0, 1, 0, 0], device=device, dtype=torch.float)]*batch_size).unsqueeze(0)
        dec_input = torch.cat([sos, x[:-1, :, :]], 0)
        h0, c0 = torch.split(torch.tanh(self.model.decoder.fc_hc(z)),
                             self.model.decoder.dec_hidden_size, 1)
        hidden_cell = (h0.unsqueeze(0).contiguous(),
                       c0.unsqueeze(0).contiguous())

        (pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy,
         q), _ = self.model.decoder(dec_input, z, hidden_cell)

        zero_out = 1 - x[:, :, 4]
        Ls = ls(x[:, :, 0], x[:, :, 1],
                pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, zero_out)
        Lp = lp(x[:, :, 2], x[:, :, 3],
                x[:, :, 4], q)
        Lr = Ls + Lp
        Lkl = lkl(mu, sigma_hat)

        step = step_in_epoch + self.step_per_epoch * self.epoch
        eta = 1.0 - (1.0 - self.eta_min) * self.R**step
        loss = Lr + self.wkl * eta * Lkl
        return loss, Ls, Lp, Lr, Lkl, self.wkl * eta * Lkl


def ls(x, y, pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, zero_out):
    Nmax = x.shape[0]
    batch_size = x.shape[1]

    pdf_val = torch.sum(pi * pdf_2d_normal(x, y, mu_x, mu_y,
                                           sigma_x, sigma_y, rho_xy), dim=2)

    return -torch.sum(zero_out * torch.log(pdf_val + 1e-5)) \
        / (Nmax * batch_size)


def lp(p1, p2, p3, q):
    p = torch.cat([p1.unsqueeze(2), p2.unsqueeze(2), p3.unsqueeze(2)], dim=2)
    return -torch.sum(p*torch.log(q + 1e-4)) \
        / (q.shape[0] * q.shape[1])


def lkl(mu, sigma, KLmin=0.2):
    lkl = -torch.sum(1+sigma - mu**2 - torch.exp(sigma)) \
        / (2. * mu.shape[0] * mu.shape[1])

    KLmin = torch.tensor(KLmin, device=device)

    return torch.max(lkl, KLmin)


def pdf_2d_normal(x, y, mu_x, mu_y, sigma_x, sigma_y, rho_xy):
    x = x.unsqueeze(2)
    y = y.unsqueeze(2)
    norm1 = x - mu_x
    norm2 = y - mu_y
    sxsy = sigma_x * sigma_y

    z = (norm1/(sigma_x + 1e-4))**2 + (norm2/(sigma_y + 1e-4))**2 -\
        (2. * rho_xy * norm1 * norm2 / (sxsy + 1e-4))

    neg_rho = 1 - rho_xy**2
    result = torch.exp(-z/(2.*neg_rho + 1e-5))
    denom = 2. * np.pi * sxsy * torch.sqrt(neg_rho + 1e-5) + 1e-5
    result = result / denom
    return result
