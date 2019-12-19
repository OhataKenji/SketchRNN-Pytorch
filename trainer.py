import numpy as np
import torch
import torch.optim as optim
from utils import strokes2rgb
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer():
    def __init__(self, model, data_loader, tb_writer, checkpoint_dir=None, learning_rate=0.0001, wkl=1.0, clip_val=1.0):
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
        self.mininum_loss = -0.2
        # TODO plot Decoder Graph
        inputs = (self.data_loader.dataset[0])[0].unsqueeze(1)
        self.tb_writer.add_graph(self.model.encoder, inputs)
        #z, _, _ = self.model.encoder(inputs)
        #self.tb_writer.add_graph(self.model.decoder, (inputs, z))

    def train(self, epoch):
        for e in range(epoch):
            self.epoch += 1
            self.tb_writer.add_scalar('progress/epoch', self.epoch, self.epoch)

            x = None
            for x, Ns in tqdm(self.data_loader, ascii=True):
                x = x.permute(1, 0, 2)
                self.train_on_batch(x, Ns)

            # TODO fix proper batch to calculate loss
            with torch.no_grad():
                loss = self.loss_on_batch(x, Ns)
                self.tb_writer.add_scalar("loss/train", loss[0], self.epoch)
                self.tb_writer.add_scalar("loss/train/Ls", loss[1], self.epoch)
                self.tb_writer.add_scalar("loss/train/Lp", loss[2], self.epoch)
                self.tb_writer.add_scalar("loss/train/Lr", loss[3], self.epoch)
                self.tb_writer.add_scalar(
                    "loss/train/Lkl", loss[4], self.epoch)

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

    def train_on_batch(self, x, Ns):
        self.model.encoder.train()
        self.model.encoder.zero_grad()
        self.model.decoder.train()
        self.model.decoder.zero_grad()

        loss, _, _, _, _ = self.loss_on_batch(x, Ns)
        loss.backward()

        torch.nn.utils.clip_grad_value_(
            self.model.encoder.parameters(), self.clip_val)
        torch.nn.utils.clip_grad_value_(
            self.model.decoder.parameters(), self.clip_val)

        self.enc_opt.step()
        self.dec_opt.step()

    def loss_on_batch(self, x, Ns):
        seq_len = x.shape[0]
        batch_size = x.shape[1]

        z, mu, sigma_hat = self.model.encoder(x)

        sos = torch.stack(
            [torch.tensor([0, 0, 1, 0, 0], device=device, dtype=torch.float)]*batch_size).unsqueeze(0)
        dec_input = torch.cat([sos, x[:-1, :, :]], 0)

        zs = torch.stack([z] * seq_len)
        dec_input = torch.cat([dec_input, zs], 2)

        (pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy,
         q), _ = self.model.decoder(dec_input, z)

        zero_out = 1 - x[:, :, 4]
        Ls = ls(x[:, :, 0], x[:, :, 1],
                pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, zero_out)
        Lp = lp(x[:, :, 2], x[:, :, 3],
                x[:, :, 4], q)
        Lr = Ls + Lp

        Lkl = lkl(mu, sigma_hat)
        loss = Lr + self.wkl * Lkl
        return loss, Ls, Lp, Lr, Lkl


def ls(x, y, pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, zero_out):
    Nmax = x.shape[0]
    batch_size = x.shape[1]

    pdf_val = torch.sum(pi * pdf_2d_normal(x, y, mu_x, mu_y,
                                           sigma_x, sigma_y, rho_xy), dim=2)

    return -torch.sum(zero_out * torch.log(pdf_val + 1e-5)) \
        / (Nmax * batch_size)


def lp(p1, p2, p3, q):
    p = torch.cat([p1.unsqueeze(2), p2.unsqueeze(2), p3.unsqueeze(2)], dim=2)
    return -torch.sum(p*torch.log(q + 1e-5)) \
        / (q.shape[0] * q.shape[1])


def lkl(mu, sigma):
    return -torch.sum(1+sigma - mu**2 - torch.exp(sigma)) \
        / (2. * mu.shape[0] * mu.shape[1])


def pdf_2d_normal(x, y, mu_x, mu_y, sigma_x, sigma_y, rho_xy):
    x = x.unsqueeze(2)
    y = y.unsqueeze(2)
    norm1 = x - mu_x
    norm2 = y - mu_y
    sxsy = sigma_x * sigma_y

    z = (norm1/(sigma_x + 1e-5))**2 + (norm2/(sigma_y + 1e-5))**2 -\
        (2. * rho_xy * norm1 * norm2 / (sxsy + 1e-5))

    neg_rho = 1 - rho_xy**2
    result = torch.exp(-z/(2.*neg_rho + 1e-5))
    denom = 2. * np.pi * sxsy * torch.sqrt(neg_rho) + 1e-5
    result = result / denom
    return result
