import os
import time
import math
import string
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
import torch
from torch import nn
import torchvision
import torch.nn.functional as F
from .tokenizer import Tokenizer
from .dataset import IAM_Dataset_line
from .model import flor,flor_lstm,flor_attention
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from tensorboardX import SummaryWriter

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    ax.plot(points)
    plt.show()

def trainIters(model, train_dataset,valid_dataset,tokenizer, batch_size=16, epochs=300, accumlate_step=1, learning_rate=1e-4, max_seq_len=128,name="lstm"):
    writer = SummaryWriter(log_dir="logs/encoder_decoder_{}".format(name))
    model.train()
    start = time.time()
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    n_iters = len(train_dataloader)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.2,patience=10)
    criterion = nn.CTCLoss()

    batch_loss = 0
    update_step = 0
    for epoch in range(epochs):
        model.train()
        for iter, (x, y, target_lengths) in enumerate(train_dataloader):

            input_lengths = torch.full(size=(x.size(0),), fill_value=max_seq_len, dtype=torch.long).to(device)
            logits = model(x.to(device))
            loss = criterion(logits.permute(1,0,2).log_softmax(2), y.to(device), input_lengths.to(device), target_lengths.to(device))
            #loss = criterion(logits.reshape(-1, logits.size(-1)).log_softmax(-1), y.reshape(-1))
            batch_loss += loss
            if iter % accumlate_step == 0:
                batch_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                mean_loss = batch_loss / accumlate_step
                print("*" * 100)
                print('%s (%d %d%%) loss: %.4f' % (timeSince(start, (iter + 1) / n_iters),
                                                   iter, iter / n_iters * 100, mean_loss))

                if update_step%10 ==0:
                    topv, topi = logits.topk(1)
                    for i in range(5):
                        pred = topi.squeeze()[i].data.cpu().numpy()

                        tar = y.squeeze()[i].data.cpu().numpy()
                        pred_sen = tokenizer.decode(pred)
                        tar_sen = tokenizer.decode(tar)
                        print()
                        print("target:", tar_sen)
                        print("predict:", pred_sen)
                        with open("logs/train_{}.txt".format(name),"a") as f:
                            f.write("target:"+ tar_sen)
                            f.write("\n")
                            f.write("predict:" + pred_sen)
                            f.write("\n")

                    writer.add_scalar("train loss", mean_loss, update_step)
                update_step += 1
                batch_loss = 0
        if epoch%10 == 0:
            torch.save({
                'model': model.state_dict()
            }, "logs/handwritten_model_{}_{}.pt".format(name,epoch))

        model.eval()
        valid_loss = 0
        valid_step = 0
        with torch.no_grad():
            for iter, (x, y, target_lengths) in enumerate(valid_dataloader):
                input_lengths = torch.full(size=(x.size(0),), fill_value=max_seq_len, dtype=torch.long).to(device)
                logits = model(x.to(device))
                loss = criterion(logits.permute(1, 0, 2).log_softmax(2), y.to(device), input_lengths.to(device),
                                 target_lengths.to(device))
                # loss = criterion(logits.reshape(-1, logits.size(-1)).log_softmax(-1), y.reshape(-1))
                valid_loss += loss
                valid_step = iter
        valid_loss = valid_loss/valid_step
        scheduler.step(valid_loss)
        writer.add_scalar("valid loss", valid_loss, update_step)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    charset_base = string.printable[:95]

    #model = flor().to(device)
    #model = flor_lstm().to(device)
    model = flor_attention().to(device)
    char_tokenizer = Tokenizer(charset_base, 128)
    print(len(char_tokenizer.chars))
    dataset_path = "IAM_dataset"
    train_dataset = IAM_Dataset_line(dataset_path=dataset_path,tokenizer=char_tokenizer, phase="train", padding=128)
    valid_dataset = IAM_Dataset_line(dataset_path=dataset_path, tokenizer=char_tokenizer, phase="valid", padding=128)

    trainIters(model, train_dataset, valid_dataset, char_tokenizer, batch_size=30, epochs=1000, accumlate_step=5,
               learning_rate=1e-4, max_seq_len=128, name="attention")