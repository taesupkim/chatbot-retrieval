from __future__ import division

from keras import backend as K
from keras import initializers, regularizers
from keras.models import Model
from keras.layers import (Layer,
                          Input,
                          Activation,
                          Dense,
                          Dot,
                          LSTM,
                          Embedding)


class AttentionAggregate(Layer):
    def __init__(self,
                 units,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 **kwargs):
        super(AttentionAggregate, self).__init__(**kwargs)
        self.units = units
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

    def build(self, input_shape):
        assert len(input_shape)==3
        self.input_size = input_shape[2]

        self.W = self.add_weight(name='W',
                                 shape=(self.input_size, self.units),
                                 initializer=self.kernel_initializer,
                                 regularizer=self.kernel_regularizer)

        self.b = self.add_weight(name='b',
                                 shape=(self.units,),
                                 initializer=initializers.Constant(0.0))

        self.V = self.add_weight(name='V',
                                 shape=(self.units, 1),
                                 initializer=self.kernel_initializer,
                                 regularizer=self.kernel_regularizer)

    def call(self, inputs, mask=None):
        if mask is None:
            mask = K.ones(tuple(inputs.shape[:-1]))

        attn_weight = K.tanh(K.dot(inputs, self.W) + self.b)
        attn_weight = K.sum(K.dot(attn_weight, self.V), axis=-1)

        attn_weight = K.exp(attn_weight - K.max(attn_weight, axis=1, keepdims=True))*mask
        attn_weight = attn_weight/K.sum(attn_weight, axis=1, keepdims=True)

        return K.sum(K.expand_dims(attn_weight, axis=-1)*inputs, axis=1)

    def compute_output_shape(self, input_shape):
        return tuple((input_shape[0], input_shape[2]))


def baseline_model(ctx_shape,
                   ans_shape,
                   vocab_size,
                   emb_size,
                   hid_size,
                   weight_regularizer=None):

    # set context index layer
    ctx_idx_data = Input(shape=ctx_shape,
                         name='ctx_idx_layer')

    # set answer index layer
    ans_idx_data = Input(shape=ans_shape,
                         name='ans_idx_layer')

    # set embedding layer
    emb_layer = Embedding(input_dim=vocab_size+1,
                          output_dim=emb_size,
                          mask_zero=True,
                          name='emb_layer')

    # set context/answer embedding data
    ctx_emb_data = emb_layer(ctx_idx_data)
    ans_emb_data = emb_layer(ans_idx_data)

    # set LSTM encoding layer
    enc_layer = LSTM(units=hid_size,
                     return_sequences=False,
                     name='enc_layer')

    # set context/answer encoded data
    ctx_enc_data = enc_layer(ctx_emb_data)
    ans_enc_data = enc_layer(ans_emb_data)

    # convert context data
    ctx_enc_data = Dense(units=hid_size,
                         use_bias=False,
                         name='similarity_layer')(ctx_enc_data)

    # dot product between context and answer
    similarity = Dot(axes=-1)([ctx_enc_data, ans_enc_data])
    similarity = Activation('sigmoid')(similarity)

    # set model
    model = Model(inputs=[ctx_idx_data, ans_idx_data], outputs=similarity)

    return model


def attention_model(ctx_shape,
                    ans_shape,
                    vocab_size,
                    emb_size,
                    hid_size,
                    weight_regularizer=None):

    # set context index layer
    ctx_idx_data = Input(shape=ctx_shape,
                         name='ctx_idx_layer')

    # set answer index layer
    ans_idx_data = Input(shape=ans_shape,
                         name='ans_idx_layer')

    # set context sentence mask layer
    ctx_mask_data = Input(shape=ctx_shape,
                          name='ctx_mask_layer')

    # set embedding layer
    emb_layer = Embedding(input_dim=vocab_size+1,
                          output_dim=emb_size,
                          mask_zero=True,
                          name='emb_layer')

    # set context/answer embedding data
    ctx_emb_data = emb_layer(ctx_idx_data)
    ans_emb_data = emb_layer(ans_idx_data)

    # set LSTM encoding layer
    enc_layer = LSTM(units=hid_size,
                     return_sequences=True,
                     name='enc_layer')

    # set context encoded data
    ctx_enc_data = enc_layer(ctx_emb_data)
    ctx_enc_data = AttentionAggregate(units=hid_size/2,
                                      name='aggr_layer')(ctx_enc_data, ctx_mask_data)

    # set answer encoded data (only last hidden)
    ans_enc_data = enc_layer(ans_emb_data)[:, -1, :]

    # convert context data
    ctx_enc_data = Dense(units=hid_size,
                         use_bias=False,
                         name='similarity_layer')(ctx_enc_data)

    # dot product between context and answer
    similarity = Dot(axes=-1)([ctx_enc_data, ans_enc_data])
    similarity = Activation('sigmoid')(similarity)

    # set model
    model = Model(inputs=[ctx_idx_data, ans_idx_data], outputs=similarity)

    return model
