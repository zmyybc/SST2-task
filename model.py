# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 18:14:53 2022

@author: ASUS
"""

#model
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNModel(nn.Module):
    
    def __init__(self,rnn_type,ntoken,ninp,nhid,nlayers,dropout=0.5,tie_weights=False):
        super(RNNModel,self).__init__()
        self.ntoken=ntoken
        self.encoder=nn.Embedding(ntoken,ninp)
        if rnn_type in ['LSTM','GRU']:
            self.rnn=getattr(nn,rnn_type)(ninp,nhid,nlayers,dropout=dropout)
        else:
            try:
                nonlinearity={'RNN_TANH':'tanh','RNN_RELU':'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                 options are ['LSTM','GRU','RNN_TANH' or 'RNN_RELU']""")
            self.rnn=nn.RNN(ninp,nhid,nlayers,nonlinearity=nonlinearity,dropout=dropout)
        self.decoder=nn.Linear(nhid,ntoken)
        
        self.drop=nn.Dropout(dropout)
        
        self.init_weights()
        
        self.rnn_type=rnn_type
        self.nhid=nhid
        self.nlayers=nlayers
    
    def init_weights(self):
        initrange=0.1
        nn.init.uniform_(self.encoder.weight,-initrange,initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight,-initrange,initrange)
    
    def forward(self,input,hidden):
        emb=self.drop(self.encoder(input))
        output,hidden=self.rnn(emb,hidden)
        output=self.drop(output)
        decoded=self.decoder(output)
        decoded=decoded.view(-1,self.ntoken)
        return decoded,hidden
    
    def  init_hidden(self,bsz):
        weight=next(self.parameters())
        if self.rnn_type =='LSTM':
            return (weight.new_zeros(self.nlayers,bsz,self.nhid),
                    weight.new_zeros(self.nlayers,bsz,self.nhid))
        else:
            return weight.new_zeros(self.nlayers,bsz,self.nhid)
                          