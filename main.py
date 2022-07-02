# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 21:15:47 2022

@author: ASUS
"""

#Main.py
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import data
import model,h

parser = argparse.ArgumentParser(description='PyTorch Wikittext-2 RNN/LSTM/GRU/Transformer Lan')
parser.add_argument('--data',type=str,default='./data/wikitext-2',help='location of the data corpus')
parser.add_argument('--model',type=str,default='LSTM',help='type of network (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
parser.add_argument('--emsize',type=int,default=200,help='size of the word embeddings')
parser.add_argument('--nhid',type=int,default=200,help='numbers of hidden units per layer')
parser.add_argument('--nlayers',type=int,defualt=2,help='the number of layers')
parser.add_argument('--lr',type=float,default=20,help='initial learning rate')
parser.add_argument('--clip',type=float,default=0.25,help='gradient clipping')
parser.add_argument('--epochs',type=int,default=40,help='upper epoch limit')
parser.add_argument('--batch_size',type=int,default=20,metavar='N',help='batch size')
parser.add_argument('--bptt',type=int,default=35,help='sequence lenth')
parser.add_argument('--dropout',type=float,default=0.2,help='dropout applied to layers')
parser.add_argument('--tied',action='store_true',help='tie the word embedding and softmax weights')
parser.add_argument('--seed',type=int,default=1111,help='random seed')
parser.add_argument('--cuda',action='store_true',help='path to save the final model')
parser.add_argument('--nhead',type=int,default=2,help='the number of heads in the encoder/decoder ')
parser.add_argument('--dry-run',actiion='store_true',help='verify the code and model')
args=parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("")

device=torch.device("cuda" if args.cuda else "cpu")

corpua=data.Corpus(args.data)

def batchify(data,bsz):
    
    nbatch=data.size(0)//bsz
    
    data=data.narrow(0,0,nbatch*bsz)
    
    data=data.view(bsz,-1).t().contiguous()
    return data.to(device)

eval_batch_size=10
train_data=batchify(corpus.train,args.batch_size)
val_data=batchify(corpus.valid,eval_batch_size)
test_data=batchify(corpus.test,eval_batch_size)

ntokens=len(corpus.dictionary)
if args.model =='LSTM':
    model=model.RNNModel(args.model,ntokens,args.emsize,args.nhid,args.nlayers,args.dropout)

criterion=nn.CrossEntropyLoss()

def repackage_hidden(h):
    
    if isinstance(h,torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def get_batch(source,i):
    seq_len=min(arg.bptt,len(source)-1-i)
    data=source[i:i+seq_len]
    target=source[i+1:i+1+seq_len].view(-1)
    return data,target

def evaluate(data_source):
    model.eval()
    total_loss=0.
    ntokens=len(corpus.dictionary)
    hidden=model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0,data_source.size(0)-1,args.bptt):
            data,targets=get_batch(data_source,i)
            output,hidden=model(data,hidden)
            hidden=repackage_hidden(hidden)
            total_loss+=len(data)*criterion(output,targets).item()
    return total_loss/(len(data_source)-1)

def train():
    model.train()
    total_loss=0
    start_time=time.time()
    ntokens=len(corpus.dictionary)
    hidden=model.init_hidden(args.batch_size)
    for batch,i in enumerate(range(0,train_data.size(0)-1,args.bptt)):
        data,targets=get_batch(train_data,i)
        model.zero_grad()
        hidden=repackage_hidden(hidden)
        output,hidden=model(data,hidden)
        loss=criterion(output,targets)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm(model.parameters(),args.clip)
        
        for p in model.parameters():
            p.data.add_(p.grad,alpha=-lr)
        
        total_loss+=loss.item()
        if batch % args.log_interval==0 and batch > 0:
            cur_loss=total_loss/args.log_interval
            elapsed=time.time()-start_time
            print('| epoch')
            total_loss=0
            start_time=time.time()
        if args.dry_run:
            break

lr=args.lr
best_val_loss=None


try:
    for epoch in range(1,args.epochs+1):
        epoch_start_time=time.time()
        train()
        val_loss=evaluate(val_data)
        print('-'*89)
        print()
        print('-'*89)
        
        if not best_val_loss or val_loss<best_val_loss:
            with open(args.save,'wb')as f:
                torch.save(model,f)
            best_val_loss=val_loss
        else:
            lr/=4.0
except KeyboardInterrupt:
    print('-'*89)
    print('Exiting from training early')
    
with open(args.save,'rb') as f:
    model=torch.load(f)
    
    if args.model in['RNN_TANH','RNN_RELU','LSTM','GRU']:
        model.rnn.flatten_parameters()
        
test_loss=evaluate(test_data)
print('='*89)
print('End')
print('='*89)