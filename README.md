# Robo-Trader-IA-ReinforceAgent


Robo para daytrade no win da bovespa.

[Aprendizagem por refor�o(RL) � uma estrutura geral onde os agentes aprendem a realizar a��es em um ambiente de forma a maximizar uma recompensa. Os dois componentes principais s�o o ambiente, que representa o problema a ser resolvido, e o agente, que representa o algoritmo de aprendizagem.](https://www.tensorflow.org/agents/tutorials/0_intro_rl?hl=pt-br#copyright_2018_the_tf-agents_authors "tf-agents, retirado 15/12/2020")

Neste caso, o ambiente � a bolsa de valores brasileira, os estados s�o os dados OHLC retirados do time-frame de 1 min dos ultimos 6 anos, as a��es s�o de compra, venda e obseva��o.O agente � uma rede neural que busca otimizar a politica. E por fim, as recompensas s�o os retornos das opera��es em pontos. Esse rob� utiliza t�cnicas de aprendizado por refor�o com tf-agents, ap�s o treinamento � criado uma rede socket para receber e enviar dados com a plataforma Metatrader.

#### Etapas de testes para encontrar o melhor resultado

Separa��o dos dados em pacotes, o per�odo de dados � o dia 11/12/2020.


c�digo em:

  * Tensor-flow
  * Tf-agents
  * Python
  * Mql5