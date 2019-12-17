import torch
from dataset import To5vStrokes, V5Dataset
from model import SketchRNN
from trainer import Trainer
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

data_path = Path.home() / 'MyDatasets/Sketches/apple/train.npy'
dataset = V5Dataset(str(data_path), To5vStrokes(), pre_scaling=True)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=100,shuffle=True)

log_dir = Path.home() / 'MLLogs/SketchRNN/pytorch/apple/testlogs/3'
tb_writer = SummaryWriter(log_dir)

checkpoint_dir = Path.home() / 'MLLogs/SketchRNN/pytorch/apple/testcheckpoints/'
model = SketchRNN()
trainer = Trainer(model, dataloader, tb_writer, checkpoint_dir)

trainer.train(epoch=3000)
tb_writer.close()
