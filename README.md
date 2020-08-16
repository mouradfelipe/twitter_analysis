# Twitter Sentiment Analysis

Projeto desenvolvido na prática de CT-213 pelo grupo:

* Adriano Soares
* Adrisson Samersla
* Felipe Mourad

## Inicialização

Primeiramente, garanta que bibliotecas como tensorflow, matplotlib e wordcloud (caso queira rodar o `print.py`) estejam instaladas.

Depois, para executar e visualizar os dados relativos a Trump e Biden basta rodar `python main.py`


## Pasta results

Temos diversas redes neurais previamente treinadas, com learning rates distintas. A estrutura é sempre a mesma

`lr<valor da learning rate>_neural_network_<estrutura>.h5`


## Pasta dataset

 Para utilizar os dados basta obtê-los da pasta dataset. Os arquivos foram descritos abaixo:
 
 * train.csv : Arquivo csv utilizado para treinamento da rede com seus devidos labels divididos em negative,neutral e positive.
 * test.csv  : Arquivo csv utilizado para teste de validação da rede com seus devidos labels divididos em negative,neutral e positive.
 * trump.csv : Arquivo csv obtido pela API do Twitter utilizando postagens relacionadas ao candidato a presidencia Donald Trump.
 * biden.csv : Arquivo csv obtido pela API do Twitter utilizando postagens relacionadas ao candidato a presidencia Joe Biden.
 
 Há ainda um dataset que não foi commitado. Trata-se do conjunto de dados pré-treinados do GloVe, disponível no link: http://nlp.stanford.edu/data/glove.twitter.27B.zip
 
## Breve descrição a respeito do código

`preprocess.py`  
 
 Responsável por realizar todo o processamento dos dados, adaptando-os para os inputs das redes neurais. Processamentos realizados: filtragem de colunas, construção de tokens das palavras, transformações dos dataframes em arrays.
 
 
`text_interpreter_nn.py`
 
 Classe que gera a arquitetura de rede neural básica RNN; entretanto, com o método `insert_emoji_feature`, ela é incrementada com algumas gírias recorrentes utilizadas no Twitter. A arquitetura gerada por esta classe gerou resultados satisfatórios. Seu treinamento ocorreu no arquivo `train_extraction.py`.
 
 
`print.py`
 
 Arquivo desenvolvido para gerar exibição de palavras no formato de WordCloud, como no exemplo dado a seguir obtido no link:
 https://i.pinimg.com/originals/51/72/6b/51726bc2b3d6fbc5b6e019d7d6c67c6b.jpg
 
 ![image](https://sebastianraschka.com/images/blog/2014/twitter-wordcloud/my_twitter_wordcloud_2_small.jpg)
 
`train_*.py`
 
 Os arquivos train*.py foram utilizados para treinar a rede utilizando diversas metodologias distintas de redes neurais. Sendo descritas abaixo.
 
 * `train_embedding.py`: Treinamento utilizando apenas o embedding. Obs: As demais também usaram embedding, porém com outros artifícios (lista incremental).
 * `train_extraction.py`: Utilizou o conceito de feature extraction para ajudar a aumentar a acurácia da rede.
 * `train_glove.py`: Utilizou o conceito de transfer learning carregando a arquitetura GloVe de rede neural.
 * `train_crnn.py` : Treinamento com Convolutional Recurrent Neural Network.

 No relatório, realizamos a comparação entre estas arquiteturas. As que apresentaran melhores desempenhos foram GloVe e Extraction.
 
 `generate_images.py`
  
  Arquivo responsável por gerar o benchmark entre as redes, sendo de extrema importância no projeto para selecionar qual a arquitetura de rede neural a utilizar-se para obter o melhor desempenho.
 
 `utils.py`
 
 Arquivo que contém algumas funções úteis que foram utilizadas no código.
