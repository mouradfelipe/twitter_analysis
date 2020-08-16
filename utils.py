from tensorflow.keras.models import model_from_json
from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt

def save_model_to_json(model, model_name):
    # serialize model to JSON
    model_json = model.to_json()
    with open(model_name + '.json', 'w') as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(model_name + '.h5')


def load_model_from_json(model_name):
    json_file = open(model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_name + '.h5')
    return loaded_model


def emoji_reader(text):
    list_of_words = text.split(' ')
    return emoji_dict(list_of_words)


def get_emojis():
    return ["<3", 'smh', 'lol', 'ahhh', 'xD', 'hahaha', 'lmfao']


def emoji_dict(list_of_words):
    list_emojis = get_emojis()
    list = [0]*len(list_emojis)
    for word in list_of_words:
        for emoji in list_emojis:
            if emoji in word:
                index = list_emojis.index(emoji)
                list[index] += 1

    return dict(zip(list_emojis, list))


def plot_word_cloud(series, file_name=''):
    comment_words = '' 
    stopwords = set(STOPWORDS) 

    for val in series.tolist():
        # typecaste each val to string
        val = str(val) 
    
        # split the value 
        tokens = val.split() 
        
        # Converts each token into lowercase 
        for i in range(len(tokens)): 
            tokens[i] = tokens[i].lower() 
        
        comment_words += " ".join(tokens)+" "
    
    wordcloud = WordCloud(width=800, height=800,
                          background_color='white',
                          stopwords=stopwords,
                          min_font_size=10).generate(comment_words)
  
    # plot the WordCloud image                        
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad=0)
    if file_name:
        file_name += '.png'
        plt.savefig(file_name, format='png')
    plt.show() 

def print_twitter_sentiment(avr_list,candidate):
    positive = avr_list[2]
    neutral = avr_list[1]
    negative = avr_list[0]
    positive_percentage = round(positive*100,2)
    negative_percentage = round(negative*100,2)
    neutral_percentage = round(neutral*100,2)
    print("\n|    Predição ",candidate)
    print("|   por comentários no Twitter")
    print("| Comentários Positivos : ",positive_percentage,"%")
    print("| Comentários Neutros   : ",neutral_percentage,"%")
    print("| Comentários Negativos : ",negative_percentage,"%\n")
