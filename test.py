import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img
import matplotlib.pyplot as plt


import vision_part1
import text_part2
import fusion_part3


# CONFIGURATION
MODEL_FILE = "modele_fusion.h5"
TOKEN_FILE = "Flickr8k.token.txt"
IMAGE_DIR = "Flicker8k_Dataset"

def load_resources():
    print("--- Chargement des ressources ---")

    # 1. Reconstruire le Tokenizer
    descriptions = vision_part1.load_flickr_descriptions(TOKEN_FILE)
    clean_desc = text_part2.clean_descriptions(descriptions)
    all_sentences = [desc for desc_list in clean_desc.values() for desc in desc_list]
    
    tokenizer = text_part2.create_tokenizer(all_sentences)
    vocab_size = len(tokenizer.word_index) + 1
    max_length = max(len(d.split()) for d in all_sentences)
    print(f"Tokenizer prêt (Vocab: {vocab_size} mots)")

    # 2. Charger ResNet
    cnn_model = vision_part1.build_cnn_encoder()
    print("Encodeur Image (ResNet) chargé")
    dummy_matrix = np.zeros((vocab_size, 300))

    fusion_model = fusion_part3.build_fusion_model(
        vocab_size=vocab_size,
        max_length=max_length,
        embedding_dim=300,
        embedding_matrix=dummy_matrix,
        n_classes=len(vision_part1.CLASSES)
    )
    
    # Maintenant que l'architecture est identique (gelée), on peut charger les poids
    fusion_model.load_weights(MODEL_FILE)
    print("Modèle de Fusion chargé avec succès !")

    return tokenizer, max_length, cnn_model, fusion_model

def predict_multimodal(image_path, text_input, tokenizer, max_length, cnn_model, fusion_model):
    """
    Fait une prédiction à partir d'une image ET d'un texte.
    """
    print(f"\nTraitement de l'image : {os.path.basename(image_path)}...")
    
    # A. Préparation Image
    img_feature = vision_part1.extract_image_features(cnn_model, image_path)
    img_feature = np.expand_dims(img_feature, axis=0)

    # B. Préparation Texte
    seq = tokenizer.texts_to_sequences([text_input])
    padded = pad_sequences(seq, maxlen=max_length, padding='post')

    # C. Prédiction
    pred_probs = fusion_model.predict([img_feature, padded], verbose=0)[0]

    # D. Affichage
    classes = vision_part1.CLASSES
    
    print(f"\nRÉSULTATS POUR : \"{text_input}\"")
    print("-" * 40)
    print(f"{'CLASSE':<12} | {'SCORE':<10} | {'DÉCISION'}")
    print("-" * 40)
    
    results = list(zip(classes, pred_probs))
    results.sort(key=lambda x: x[1], reverse=True) 

    for cls, prob in results:
        decision = "OUI" if prob > 0.5 else "NON"
        print(f"{cls:<12} | {prob:.2%}      | {decision}")

    # E. Montrer l'image
    try:
        img = load_img(image_path)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Input: {text_input}")
        plt.show()
    except:
        pass

if __name__ == "__main__":
    # 1. Tout charger
    tokenizer, max_len, cnn, model = load_resources()

    # 2. Choisir une image au hasard
    import random
    all_files = os.listdir(IMAGE_DIR)
    
    # On filtre pour ne garder que les fichiers qui finissent par .jpg
    valid_images = [f for f in all_files if f.endswith(".jpg") and ":" not in f]
    
    if len(valid_images) == 0:
        print(f"Erreur : Aucune image .jpg trouvée dans {IMAGE_DIR}")
        exit()

    random_file = random.choice(valid_images)
    image_path_test = os.path.join(IMAGE_DIR, random_file)

    # 3. Tester snas phrase
    phrase_test = ""
    
    # 4. Lancer la prédiction
    predict_multimodal(image_path_test, phrase_test, tokenizer, max_len, cnn, model)