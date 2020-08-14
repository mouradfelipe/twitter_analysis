from tensorflow.keras.models import model_from_json

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
    return  ["<3", 'kkk', 'lol','ahhh','xD','hahaha']

def emoji_dict(list_of_words):
    list_emojis = get_emojis()
    list = [0]*len(list_emojis)
    for word in list_of_words:
        for emoji in list_emojis:
            if emoji in word:
                index = list_emojis.index(emoji)
                list[index]+=1
    list = [[item] for item in list] # Adaptei para rodar no pandas

    return dict(zip(list_emojis,list))
