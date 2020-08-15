import pickle

import matplotlib.pyplot as plt


fig_format = 'png'

with open('./results/embedding.data', 'rb') as file:
    embedding_data = pickle.load(file)

with open('./results/glove.data', 'rb') as file:
    glove_data = pickle.load(file)


plt.plot(embedding_data['loss'])
plt.plot(embedding_data['val_loss'])
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title('Cost Function')
plt.grid()
plt.savefig('./results/loss' + '.' + fig_format, format=fig_format)


plt.plot(embedding_data['accuracy'])
plt.plot(embedding_data['val_accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.grid()
plt.savefig('./accuracy' + '.' + fig_format, format=fig_format)

extraction_data = {}
crnn_data = {}

