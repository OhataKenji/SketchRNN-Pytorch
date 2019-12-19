import torch
from dataset import To5vStrokes, V5Dataset
from model import SketchRNN
from trainer import Trainer
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

data_path = Path.home() / 'MyDatasets/Sketches/apple/train.npy'
dataset = V5Dataset(str(data_path), To5vStrokes(max_len=80), pre_scaling=True)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=100, shuffle=True)

log_dir = Path.home() / 'MLLogs/SketchRNN/pytorch/apple/testlogs/t6'
tb_writer = SummaryWriter(log_dir)

checkpoint_dir = Path.home() / 'MLLogs/SketchRNN/pytorch/apple/testcheckpoints/'
model = SketchRNN(enc_hidden_size=256, dec_hidden_size=512,
                  Nz=128, M=20, dropout=0.1)
trainer = Trainer(model, dataloader, tb_writer,
                  checkpoint_dir, learning_rate=0.001)

trainer.train(epoch=300000)
tb_writer.close()
