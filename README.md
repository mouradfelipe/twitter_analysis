# Twitter Sentiment Analysis

Projeto desenvolvido na prática de CT-213 pelo grupo:

* Adriano Soares
* Felipe Mourad
* Adrisson Samersla

## Inicialização

Primeiramente, garanta que bibliotecas como tensorflow, matplotlib e wordcloud (caso queira rodar o `print.py`)

Depois, para executar e visualizar os dados relativos a Trump e Biden basta rodar `python main.py`

OBS: Para calcular os dados médios de aprovação do Trump, favor commitar as linhas relativas ao Joe Biden, caso contrário fazer o oposto.

## Pasta results

Temos diversas redes neurais previamente treinadas, com learning rates distintas. A estrutura é sempre a mesma

`lr<valor da learning rate>_neural_network_<estrutura>.h5`


## Pasta dataset

 Para utilizar os dados basta obtê-los da pasta dataset. Os arquivos foram descritos abaixo:
 
 * train.csv : Arquivo csv utilizado para treinamento da rede com seus devidos labels divididos em negative,neutral e positive.
 * test.csv  : Arquivo csv utilizado para teste de validação da rede com seus devidos labels divididos em negative,neutral e positive.
 * trump.csv : Arquivo csv obtido pela API do Twitter utilizando postagens relacionadas ao candidato a presidencia Donald Trump.
 * biden.csv : Arquivo csv obtido pela API do Twitter utilizando postagens relacionadas ao candidato a presidencia Joe Biden.
 
 
## Breve descrição a respeito do código

`preprocess.py`  
 
 Responsável por realizar todo o processamento dos dados adaptando-os para os inputs das redes neurais e para adaptar os dataframes seja através de filtragem de colunas, construção de tokens das palavras ou transformações dos dataframes em arrays.
 
 
`text_interpreter_nn.py`
 
 Classe que gera a arquitetura de rede neural básica CRNN, entretanto com o método `insert_emoji_feature` ela incrementa com algumas gírias recorrentes utilizadas no Twitter. A arquitetura gerada por esta classe gerou resultados satisfatórios. Seu treinamento ocorreu no arquivo `train_extraction.py`.
 
 
`print.py`
 
 Arquivo desenvolvido para gerar exibição de palavras no formato de WordCloud, como no exemplo dado a seguir obtido no link:
 https://i.pinimg.com/originals/51/72/6b/51726bc2b3d6fbc5b6e019d7d6c67c6b.jpg
 
 ![image](https://sebastianraschka.com/images/blog/2014/twitter-wordcloud/my_twitter_wordcloud_2_small.jpg)
 
`train_*.py`
 
 Os arquivos train*.py foram utilizados para treinar a rede utilizando diversas metodologias distintas de redes neurais. Sendo descritas abaixo.
 
 * `train_crnn.py` : Treinamento com Convolutional Recurrent Neural Network.
 * `train_embedding.py`: Treinamento utilizando apenas o embedding. Obs: As demais também usaram embedding mas outros artifícios também.
 * `train_extraction.py`: Utilizou o conceito de extraction features para ajudar a aumentar a acurácia da rede.
 * `train_glove.py`: Utilizou apenas o conceito de transfer learning carregando a arquitetura GloVe de rede neural.

 No relatório, realizamos a comparação entre estas arquiteturas, as que apresentaran melhores desempenhos foram a GloVe e a Extraction.
 
 `generate_images.py`
  
  Arquivo responsável por gerar o benchmark entre as redes, sendo de extrema importância no projeto para selecionar qual a arquitetura de rede neural utilizar a fim de obter o melhor desempenho.
 
 `utils.py`
 
 Arquivo que contém algumas funções uteis e que foram utilizadas no código.
