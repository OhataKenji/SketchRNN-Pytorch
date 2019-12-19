import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SketchRNN():
    def __init__(self, enc_hidden_size=512, dec_hidden_size=2048, Nz=128, M=20, dropout=0.1):
        self.encoder = Encoder(enc_hidden_size, Nz, dropout).to(device)
        self.decoder = Decoder(dec_hidden_size, Nz, M, dropout).to(device)

    # TODO batch reconstruction
    def reconstruct(self, S):
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            Nmax = S.shape[0]
            batch_size = S.shape[1]
            s_i = torch.stack(
                [torch.tensor([0, 0, 1, 0, 0], device=device, dtype=torch.float)] * batch_size, dim=0).unsqueeze(0)
            output = s_i  # dummy
            z, _, _ = self.encoder(S)
            for i in range(Nmax):
                s_i_z = torch.cat([s_i, z.unsqueeze(0)], 2)
                (pi, mu_x, mu_y, sigma_x, sigma_y,
                 rho_xy, q), _ = self.decoder(s_i_z, z)
                s_i = self.sample_next(
                    pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q)
                output = torch.cat([output, s_i], dim=0)
                if output[-1, 0, 4] == 1:
                    break

            output = output[1:, :, :]  # remove dummy
            return output

    def sample_next(self, pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q):
        pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q =\
            pi[0, 0, :], mu_x[0, 0, :], mu_y[0, 0, :], sigma_x[0,
                                                               0, :], sigma_y[0, 0, :], rho_xy[0, 0, :], q[0, 0, :]
        mu_x, mu_y, sigma_x, sigma_y, rho_xy =\
            mu_x.cpu().numpy(), mu_y.cpu().numpy(), sigma_x.cpu(
            ).numpy(), sigma_y.cpu().numpy(), rho_xy.cpu().numpy()
        M = pi.shape[0]
        # offset
        idx = np.random.choice(M, p=pi.cpu().numpy())
        mean = [mu_x[idx], mu_y[idx]]
        cov = [[sigma_x[idx] * sigma_x[idx], rho_xy[idx] * sigma_x[idx]*sigma_y[idx]],
               [rho_xy[idx] * sigma_x[idx]*sigma_y[idx], sigma_y[idx] * sigma_y[idx]]]
        xy = np.random.multivariate_normal(mean, cov, 1)
        xy = torch.from_numpy(xy).float().to(device)

        # pen
        p = torch.tensor([0, 0, 0], device=device, dtype=torch.float)
        idx = np.random.choice(3, p=q.cpu().numpy())
        p[idx] = 1.0
        p = p.unsqueeze(0)

        return torch.cat([xy, p], dim=1).unsqueeze(0)


class Encoder(nn.Module):
    def __init__(self, enc_hidden_size=512, Nz=128, dropout=0.1):
        super().__init__()
        self.encoder_rnn = nn.LSTM(
            5, enc_hidden_size, dropout=dropout, bidirectional=True)
        self.fc_mu = nn.Linear(2*enc_hidden_size, Nz)
        self.fc_sigma = nn.Linear(2*enc_hidden_size, Nz)

    def forward(self, inputs):
        _, (hidden, cell) = self.encoder_rnn(inputs)
        h_forward, h_backward = torch.split(hidden, 1, 0)
        h = torch.cat([h_forward.squeeze(0), h_backward.squeeze(0)], 1)

        mu = self.fc_mu(h)
        sigma_hat = self.fc_sigma(h)
        sigma = torch.exp(sigma_hat/2.)

        N = torch.normal(torch.zeros(mu.size()),
                         torch.ones(mu.size())).to(device)
        z = mu + sigma * N
        return z, mu, sigma_hat


class Decoder(nn.Module):
    def __init__(self, dec_hidden_size=2048, Nz=128, M=20, dropout=0.1):
        super().__init__()
        self.M = M
        self.dec_hidden_size = dec_hidden_size
        self.fc_hc = nn.Linear(Nz, 2*dec_hidden_size)
        self.decoder_rnn = nn.LSTM(Nz+5, dec_hidden_size, dropout=dropout)
        self.fc_y = nn.Linear(dec_hidden_size, 6*M+3)

    def forward(self, dec_input, z, hidden_cell=None):
        if hidden_cell is None:
            # then we must init from z
            hidden, cell = torch.split(
                F.tanh(self.fc_hc(z)), self.dec_hidden_size, 1)
            hidden_cell = (hidden.unsqueeze(0).contiguous(),
                           cell.unsqueeze(0).contiguous())
        outputs, (hidden, cell) = self.decoder_rnn(dec_input, hidden_cell)
        # in training we feed the lstm with the whole input in one shot
        # and use all outputs contained in 'outputs', while in generate
        # mode we just feed with the last generated sample:
        y = self.fc_y(outputs.view(-1, self.dec_hidden_size))
        # separate pen and mixture params:
        params = torch.split(y, 6, 1)
        params_mixture = torch.stack(params[:-1])  # trajectory
        params_pen = params[-1]  # pen up/down
        # identify mixture params:
        pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy = torch.split(
            params_mixture, 1, 2)
        # preprocess params::
        len_out = dec_input.shape[0]

        pi = F.softmax(pi.transpose(0, 1).squeeze()).view(len_out, -1, self.M)
        sigma_x = torch.exp(sigma_x.transpose(
            0, 1).squeeze()).view(len_out, -1, self.M)
        sigma_y = torch.exp(sigma_y.transpose(
            0, 1).squeeze()).view(len_out, -1, self.M)
        rho_xy = torch.tanh(rho_xy.transpose(
            0, 1).squeeze()).view(len_out, -1, self.M)
        mu_x = mu_x.transpose(0, 1).squeeze(
        ).contiguous().view(len_out, -1, self.M)
        mu_y = mu_y.transpose(0, 1).squeeze(
        ).contiguous().view(len_out, -1, self.M)
        q = F.softmax(params_pen).view(len_out, -1, 3)
        return (pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q), (hidden, cell)
