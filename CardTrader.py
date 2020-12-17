# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 10:45:11 2020

@author: jrmfi
"""
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
import numpy as np
import pandas as pd
from Trade import Trade

class CardGameEnv(py_environment.PyEnvironment):

  def __init__(self,base,fim:int):
    self._action_spec = array_spec.BoundedArraySpec(
        shape=(1,), dtype=np.float32, minimum=0, maximum=2, name='action')
    self._observation_spec = array_spec.BoundedArraySpec(
        shape=(1,12), dtype=np.float32, minimum=[-20.0,-20.0,-20.0,-20.0,-20.0,-20.0,-20.0,-20.0,-20.0,-20.0,
                                                -20.0,-20.0], 
                                        maximum=[20.0,20.0,20.0,20.0,20.0,20.0,20.0,20.0,20.0,20.0,
                                                 20.0,20.0], name='observation')
    
    self._episode_ended = False
    self.base = base
    self.cont = 0
    self.fim = fim
    self.contador = 0
    self.metal = False
    self.dados1,self.dados2 = self.tratamento(base,fim)
    self.trader = Trade()
    self._state =  self.dados2.values[0].tolist()#0 # self.dados2.values[0].tolist()
    print('--------------------')
    print(self._state)
    print(type(self._state))
    print('--------------------')
  #-----------------------------------------------------------------

  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  def _reset(self):
    self._state = self.dados2.values[self.cont]  #0 # self.dados2.values[0].tolist()
    self._episode_ended = False
    return ts.restart(np.array([self._state], dtype=np.float32))

  def _step(self, action):

    if self._episode_ended:
    #    # The last action ended the episode. Ignore the current action and start
    #    # a new episode.
       self.reset()
      
    # Make sure episodes don't go on forever.
    stop = -500
    gain = 500
    if len(self.dados1)-5 <= self.cont:
        self.contador +=1
        self.trader.reset()
        print('contador: ',self.contador)
        self._episode_ended = True
        self.cont = 0
    # dados1,dados2 = self.tratamento(self.base)
    compra,venda,neg,ficha,comprado,vendido,recompensa=self.trader.agente(self.dados1.values[self.cont],action,stop,gain,0)
    if comprado or vendido:
        self.metal = True
    if self.metal and (comprado == False and vendido == False):
        self.metal = False
        self._episode_ended = True
        
    # print('dentro da classe-----------------------')
    # print('cont :',self.cont)
    # print('acao: ',action)
    # print('comprado: ',comprado)
    # print('vendido: ',vendido)
    # print('recompensa: ',recompensa)
    # # print('dados1: ',self.dados1.values[self.cont][0]) #68084.0
    # # print('dados1: ',self.dados2.values[self.cont])
    # print('episodio: ',self._episode_ended )
    # print('-------------------------------------')
    
    self.state = self.dados2.values[self.cont].tolist()
    self.cont += 1

    
      # raise ValueError('`action` should be 0 or 1.')
    reward = recompensa
    if self._episode_ended:
      self.trader.reset()
      return ts.termination(np.array([self._state], dtype=np.float32), reward)
    else:
      return ts.transition(
          np.array([self._state], dtype=np.float32), reward=reward, discount=1.0)
  #-----------------------------------------------------------------
  def tratamento(self,base,fim):
    print('****************************')
    print('tratamento: ',self.cont)
    print('****************************')
    colunas = ['Hora','dif', 'retacao +','retracao -', 'RSI',
                 'M22M44', 'M22M66', 'M66M44', 'ADX', 'ATR',
                'Momentum', 'Force']
    
    colunas1 = ['Hora', 'open', 'high', 'low', 'close']
    dados1 = pd.DataFrame(data=base[-fim:-1].values,columns=base.columns)      
    dados2 = pd.DataFrame(data=base[-fim:-1].values,columns=base.columns)
    dados1 = dados1[colunas1]
    dados2 = dados2[colunas]
    index = 0
    for i in dados2.values:
        base1 = i[0].split(':')
        dados2.at[index, 'Hora'] = float(base1[0])*100 + float(base1[1])
        index += 1
    train_mean = dados2.mean(axis=0)
    train_std = dados2.std(axis=0)
    dados2 = (dados2 - train_mean) / train_std
    return dados1,dados2
        