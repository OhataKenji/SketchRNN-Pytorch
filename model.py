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
        with torch.no_grad():
            Nmax = S.shape[0]
            batch_size = S.shape[1]
            s_i = torch.stack(
                [torch.tensor([0, 0, 1, 0, 0], device=device, dtype=torch.float)] * batch_size, dim=0).unsqueeze(0)
            output = s_i  # dummy
            z, _, _ = self.encoder(S)
            for i in range(Nmax):
                (pi, mu_x, mu_y, sigma_x, sigma_y,
                 rho_xy, q), _ = self.decoder(s_i, z)
                s_i = self.sample_next(
                    pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q)
                output = torch.cat([output, s_i], dim=0)
                if output[-1, 0, 4] == 1:
                    break

            output = output[1:, :, :]  # remove dummy
            return output

    def sample_next(self, pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q):
        pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q =\
             pi[0, 0, :], mu_x[0, 0, :], mu_y[0,0, :], sigma_x[0, 0, :], sigma_y[0, 0, :], rho_xy[0, 0, :], q[0, 0, :]
        mu_x, mu_y, sigma_x, sigma_y, rho_xy =\
            mu_x.cpu().numpy(), mu_y.cpu().numpy(), sigma_x.cpu().numpy(), sigma_y.cpu().numpy(), rho_xy.cpu().numpy()
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
        h = torch.cat((hidden[0, :, :], hidden[1, :, :]), 1)

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

    def forward(self, S, z):
        batch_size = S.shape[1]
        seq_len = S.shape[0]

        h0, c0 = torch.split(torch.tanh(self.fc_hc(z)),
                             self.dec_hidden_size, 1)
        h0c0 = (h0.unsqueeze(0).contiguous() , c0.unsqueeze(0).contiguous())

        sos = torch.stack(
            [torch.tensor([0, 0, 1, 0, 0], device=device, dtype=torch.float)]*batch_size).unsqueeze(0)
        zs = torch.stack([z] * seq_len)
        dec_input = torch.cat([sos, S[:-1, :, :]], 0)
        dec_input = torch.cat([dec_input, zs], 2)

        o, (h, c) = self.decoder_rnn(dec_input, h0c0)
        y = self.fc_y(o)

        pi = F.softmax(y[:, :, 0:self.M], dim=2)
        mu_x = y[:, :, self.M:2*self.M]
        mu_y = y[:, :, 2*self.M:3*self.M]
        sigma_x = torch.exp(y[:, :, 3*self.M:4*self.M])
        sigma_y = torch.exp(y[:, :, 4*self.M:5*self.M])
        rho_xy = torch.tanh(y[:, :, 5*self.M:6*self.M])
        q = F.softmax(y[:, :, 6*self.M:6*self.M+3], dim=2)

        return (pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q), (h, c)
