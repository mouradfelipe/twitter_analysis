import pickle

import matplotlib.pyplot as plt


fig_format = 'png'

with open('results/lr01_embedding.data', 'rb') as file:
    embedding_data = pickle.load(file)

with open('results/lr01_glove.data', 'rb') as file:
    glove_data = pickle.load(file)

with open('results/lr01_extraction.data', 'rb') as file:
    extraction_data = pickle.load(file)

with open('results/lr01_crnn.data', 'rb') as file:
    crnn_data = pickle.load(file)


# Plotting loss for training and validation datasets
def plot(data, feature, name, title, ylabel):
    plt.figure()

    train, = plt.plot(data[feature], label='Treino, lr=')
    val, = plt.plot(data['val_' + feature], label='Validação')
    plt.legend(handles=[train, val])

    plt.xlabel('Épocas')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    plt.savefig('./results/' + feature + '_' + name + '.' + fig_format, format=fig_format)


def plot_comparative(feature, name, title, ylabel):
    plt.figure()

    embed, = plt.plot(embedding_data['val_' + feature], label='Embedding')
    glove, = plt.plot(glove_data['val_' + feature], label='Glove')
    extract, = plt.plot(extraction_data['val_' + feature], label='Extraction')
    crnn, = plt.plot(crnn_data['val_' + feature], label='CRNN')
    plt.legend(handles=[embed, glove, extract, crnn])

    plt.xlabel('Épocas')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    plt.savefig('./results/' + feature + '_' + name + '.' + fig_format, format=fig_format)


plot(embedding_data, 'loss', 'embedding', 'Embedding - Função de Custo', 'Custo')
plot(glove_data, 'loss', 'glove', 'Glove - Função de Custo', 'Custo')
plot(extraction_data, 'loss', 'extraction', 'Extraction - Função de Custo', 'Custo')
plot(crnn_data, 'loss', 'crnn', 'CRNN - Função de Custo', 'Custo')

plot(embedding_data, 'accuracy', 'embedding', 'Embedding - Acurácia', 'Acurácia')
plot(glove_data, 'accuracy', 'glove', 'Glove - Acurácia', 'Acurácia')
plot(extraction_data, 'accuracy', 'extraction', 'Extraction - Acurácia', 'Acurácia')
plot(crnn_data, 'accuracy', 'crnn', 'CRNN - Acurácia', 'Acurácia')

plot_comparative('loss', 'comparative', 'Comparativo - Função de Custo', 'Custo')
plot_comparative('accuracy', 'comparative', 'Comparativo - Acurácia', 'Acurácia')
