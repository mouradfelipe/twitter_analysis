import pickle

import matplotlib.pyplot as plt


fig_format = 'png'

with open('results/lr01_embedding.data', 'rb') as file:
    embedding_data_01 = pickle.load(file)

with open('results/lr01_glove.data', 'rb') as file:
    glove_data_01 = pickle.load(file)

with open('results/lr01_extraction.data', 'rb') as file:
    extraction_data_01 = pickle.load(file)

with open('results/lr01_crnn.data', 'rb') as file:
    crnn_data_01 = pickle.load(file)


with open('results/lr001_embedding.data', 'rb') as file:
    embedding_data_001 = pickle.load(file)

with open('results/lr001_glove.data', 'rb') as file:
    glove_data_001 = pickle.load(file)

with open('results/lr001_extraction.data', 'rb') as file:
    extraction_data_001 = pickle.load(file)

with open('results/lr001_crnn.data', 'rb') as file:
    crnn_data_001 = pickle.load(file)


# Plotting loss for training and validation datasets
def plot(data_01, data_001, feature, name, title, ylabel):
    plt.figure()

    train_01, = plt.plot(data_01[feature], label='Treino, lr=0.01')
    val_01, = plt.plot(data_01['val_' + feature], label='Validação, lr=0.01')

    train_001, = plt.plot(data_001[feature], label='Treino, lr=0.001')
    val_001, = plt.plot(data_001['val_' + feature], label='Validação, lr=0.001')

    plt.legend(handles=[train_01, val_01, train_001, val_001])

    plt.xlabel('Épocas')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    plt.savefig('./results/' + feature + '_' + name + '.' + fig_format, format=fig_format)


def plot_comparative(data, feature, name, title, ylabel, prefixe):
    plt.figure()

    embedding_data = data['embedding']
    glove_data = data['glove']
    extraction_data = data['extraction']
    crnn_data = data['crnn']

    embed, = plt.plot(embedding_data['val_' + feature], label='Embedding')
    glove, = plt.plot(glove_data['val_' + feature], label='Glove')
    extract, = plt.plot(extraction_data['val_' + feature], label='Extraction')
    crnn, = plt.plot(crnn_data['val_' + feature], label='CRNN')
    plt.legend(handles=[embed, glove, extract, crnn])

    plt.xlabel('Épocas')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    plt.savefig('./results/' + prefixe + '_' + feature + '_' + name + '.' + fig_format, format=fig_format)


plot(embedding_data_01, embedding_data_001, 'loss', 'embedding', 'Embedding - Função de Custo', 'Custo')
plot(glove_data_01, glove_data_001, 'loss', 'glove', 'Glove - Função de Custo', 'Custo')
plot(extraction_data_01, extraction_data_001, 'loss', 'extraction', 'Extraction - Função de Custo', 'Custo')
plot(crnn_data_01, crnn_data_001, 'loss', 'crnn', 'CRNN - Função de Custo', 'Custo')

plot(embedding_data_01, embedding_data_001, 'accuracy', 'embedding', 'Embedding - Acurácia', 'Acurácia')
plot(glove_data_01, glove_data_001, 'accuracy', 'glove', 'Glove - Acurácia', 'Acurácia')
plot(extraction_data_01, extraction_data_001, 'accuracy', 'extraction', 'Extraction - Acurácia', 'Acurácia')
plot(crnn_data_01, crnn_data_001, 'accuracy', 'crnn', 'CRNN - Acurácia', 'Acurácia')


data_01 = {
    'embedding': embedding_data_01,
    'glove': glove_data_01,
    'extraction': extraction_data_01,
    'crnn': crnn_data_01
}
plot_comparative(data_01, 'loss', 'comparative', 'Comparativo - Função de Custo (lr=0.01)', 'Custo', 'lr01')
plot_comparative(data_01, 'accuracy', 'comparative', 'Comparativo - Acurácia (lr=0.01)', 'Acurácia', 'lr01')


data_001 = {
    'embedding': embedding_data_001,
    'glove': glove_data_001,
    'extraction': extraction_data_001,
    'crnn': crnn_data_001
}
plot_comparative(data_001, 'loss', 'comparative', 'Comparativo - Função de Custo (lr=0.001)', 'Custo', 'lr001')
plot_comparative(data_001, 'accuracy', 'comparative', 'Comparativo - Acurácia (lr=0.001)', 'Acurácia', 'lr001')
