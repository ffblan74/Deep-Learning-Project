import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os


import vision_part1
import text_part2

# Config
TOKEN_FILE = "Flickr8k.token.txt"
IMAGE_PATH_TO_TEST = "Flicker8k_Dataset/1000268201_693b08cb0e.jpg" # Mets une image qui existe
MODEL_FILE = "modele_fusion.h5" 

# Note: Pour que ça marche, il faut avoir sauvegardé le modèle à la fin du main.py


def main():
    if not os.path.exists(MODEL_FILE):
        print("Modèle non trouvé.")
        return

    #1. Recharger les outils (Tokenizer)
    print("Chargement outils")
    descriptions = vision_part1.load_flickr_descriptions(TOKEN_FILE)
    clean_desc = text_part2.clean_descriptions(descriptions)
    all_sentences = [desc for desc_list in clean_desc.values() for desc in desc_list]
    tokenizer = text_part2.create_tokenizer(all_sentences)
    max_length = max(len(d.split()) for d in all_sentences)

    #2. Charger le modèle
    model = load_model(MODEL_FILE)
    print("Modèle chargé.")

    #3. Préparer l'image (PIXELS)
    # On utilise la fonction qui renvoie (1, 224, 224, 3)
    img_input = vision_part1.load_and_preprocess_image(IMAGE_PATH_TO_TEST)
    
    #4. Préparer un texte (Dummy ou réel)
    # Le modèle a besoin de texte aussi. Pour tester juste l'image, on peut mettre un texte vide ou générique.
    # Ou alors on teste une paire Image + Texte spécifique.
    dummy_text = "<start> dog <end>"
    seq = tokenizer.texts_to_sequences([dummy_text])[0]
    seq_padded = pad_sequences([seq], maxlen=max_length, padding='post')

    # 5. Prédiction
    pred = model.predict([img_input, seq_padded])[0]
    
    print("\nPrédictions :")
    for i, cls in enumerate(vision_part1.CLASSES):
        print(f"{cls}: {pred[i]:.4f}")
        
    plt.imshow(plt.imread(IMAGE_PATH_TO_TEST))
    plt.show()

if __name__ == "__main__":
    main()