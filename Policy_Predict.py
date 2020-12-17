# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 14:28:56 2020

@author: jrmfi
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
# import tensorflow_probability as tfp
import numpy as np

# from tf_agents.specs import array_spec
# from tf_agents.specs import tensor_spec
# from tf_agents.networks import network

# from tf_agents.policies import py_policy
# from tf_agents.policies import random_py_policy
# from tf_agents.policies import scripted_py_policy

# from tf_agents.policies import tf_policy
# from tf_agents.policies import random_tf_policy
# from tf_agents.policies import actor_policy
# from tf_agents.policies import q_policy
# from tf_agents.policies import greedy_policy

from tf_agents.trajectories import time_step as ts
# import pandas as pd
# import chardet
# import tensorflow as tf
# import numpy as np
from comunica import  Comunica
tf.compat.v1.enable_v2_behavior()
# model 2
# media = np.array([ 1.35001064e+03,  1.41843972e+00,  2.31737589e+01,  2.06737589e+01,
#         5.08349468e+01,  2.21695035e+00,  5.41413121e+00, -3.19700355e+00,
#         2.94372695e+01,  8.52759752e+01,  1.00001117e+02,  8.17068867e+03])

# std = np.array([2.71505290e+02, 5.30217498e+01, 1.87897109e+01, 1.64847727e+01,
#        1.14770442e+01, 6.58121769e+01, 1.10372975e+02, 4.77192814e+01,
#        1.00903737e+01, 1.98468576e+01, 1.93246528e-01, 1.38057677e+05])
# model 3
media = np.array([ 1.33668111e+03, -5.59011180e-02,  2.32453649e+01,  2.15980320e+01,
        5.03135393e+01,  2.69175544e+00,  5.35813796e+00, -2.66638213e+00,
        3.01251349e+01,  8.47602054e+01,  1.00003427e+02, -8.22468017e+02])
std = np.array([2.62060207e+02, 5.41873032e+01, 2.07955432e+01, 2.02269798e+01,
       1.23945410e+01, 8.58785931e+01, 1.47621791e+02, 6.55143533e+01,
       1.19292385e+01, 3.07467607e+01, 2.40703765e-01, 2.07027500e+05])
# batch_size = 3
saved_policy = tf.saved_model.load('policy3')
# policy_state = saved_policy.get_initial_state(batch_size=batch_size)

HOST = ''    # Host
PORT = 8888  # Porta
R = Comunica(HOST,PORT)
s = R.createServer()

while True:
    p,addr = R.runServer(s)
    jm = np.array((p-media)/std)
    jm = np.array(jm, dtype=np.float32)
    observations = tf.constant([[jm]])
    # print(observations)
    time_step = ts.restart(observations,1)
    # print(time_step)
    action = saved_policy.action(time_step)
    previsao2 = action.action.numpy()[0]
    d3 = p[0]
    print('recebido: ',p[0])
    print('previsao: ',previsao2)
    if previsao2 == 0:
        print('Sem operacao')
    if previsao2 == 1:
        flag = "compra-{}".format(d3)
        # flag ="compra"
        print('compra: ',previsao2)
        R.enviaDados(flag,s,addr)
    if previsao2 == 2:
        flag = "venda-{}".format(d3)
        # flag = "venda"
        print('venda: ',previsao2)
        R.enviaDados(flag,s,addr)


