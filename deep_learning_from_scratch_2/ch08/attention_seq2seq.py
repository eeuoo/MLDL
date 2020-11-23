# coding: utf-8
import sys
sys.path.append('..')
from common.time_layers import *
from ch07.seq2seq import Encoder, Seq2seq
from ch08.attention_layer import TimeAttention

class AttentionEncoder(Encoder) :
    def forward(self, xs) :
        xs = self.embed.forward(xs)
        hs = self.lstm.forward(xs) # 모든 은닉상태 반환 (앞 장에서는 마지막 상태 벡터만 반환했었음)
        return hs

    def backward(self, dhs) :
        dout = self.lstm.backward(dhs)
        dout = self.embed.backward(dout)
        return dout