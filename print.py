from preprocess import load_dataframe
from utils import plot_word_cloud

file_path = './dataset/Biden.csv'
dataframe = load_dataframe(file_path)
plot_word_cloud(dataframe['text'], './results/biden')
