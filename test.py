import torch
from dataset import To5vStrokes, V5Dataset
from model import SketchRNN
from trainer import Trainer
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

data_path = Path.home() / 'MyDatasets/Sketches/apple/test.npy'
dataset = V5Dataset(str(data_path), To5vStrokes(max_len=80))
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=10, shuffle=True)

log_dir = Path.home() / 'MLLogs/SketchRNN/pytorch/apple/testlogs/1'
tb_writer = SummaryWriter(log_dir)

model = SketchRNN(enc_hidden_size=64, dec_hidden_size=64,
                  Nz=64, M=5, dropout=0.1)
trainer = Trainer(model, dataloader, tb_writer, learning_rate=0.0001)
inputs = (dataloader.dataset[0])[0].unsqueeze(1)
print(model.reconstruct(inputs))
# trainer.train(epoch=300)
tb_writer.close()
