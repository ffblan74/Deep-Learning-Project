import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# Import des outils pour un entraînement (sauvegarde auto, arrêt auto)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.sequence import pad_sequences


import vision_part1
import text_part2
import fusion_part3


# Config globale
IMAGE_DIR = "Flicker8k_Dataset"            # Dossier des images
TOKEN_FILE = "Flickr8k.token.txt"          # Fichier des descriptions
GLOVE_FILE = "glove.6B.300d.txt"           # Fichier GloVe
MAX_IMAGES = 8000  # On passe à TOUT le dataset pour un modèle fort
BATCH_SIZE = 64    # Un peu plus gros pour accélérer
EPOCHS = 50        # On laisse le temps d'apprendre

def main():
    print("Demarrage du projet Deep Learning Multi-Modal\n")

    # etape 1 : Préparation du TEXTE (Partie 2)
    print("\n[1/5] Chargement et nettoyage du texte")
    descriptions = text_part2.load_flickr_descriptions(TOKEN_FILE)
    if not descriptions:
        print("Erreur : Descriptions introuvables.")
        return

    clean_desc = text_part2.clean_descriptions(descriptions)
    
    # Rassembler toutes les phrases pour créer le vocabulaire
    all_sentences = [desc for desc_list in clean_desc.values() for desc in desc_list]
    
    tokenizer = text_part2.create_tokenizer(all_sentences)
    vocab_size = len(tokenizer.word_index) + 1
    max_length = max(len(d.split()) for d in all_sentences)
    
    print(f"      Vocabulaire : {vocab_size} mots")
    print(f"      Longueur max  : {max_length}")

    # Chargement de GloVe
    print("\n[2/5] Chargement des Embeddings GloVe")
    if os.path.exists(GLOVE_FILE):
        embedding_matrix = text_part2.load_glove_embeddings(GLOVE_FILE, tokenizer.word_index, embedding_dim=300)
    else:
        print("Fichier GloVe introuvable. On entraînera sans pré-entraînement.")
        embedding_matrix = None


    # etape 2 : Préparation des IMAGES et DATASET (Partie 1 + Boucle)
    print(f"\n[3/5] Construction du Dataset ({MAX_IMAGES} images)...")
    
    # On charge le CNN (ResNet) juste pour l'extraction
    cnn_model = vision_part1.build_cnn_encoder()
    
    image_ids = list(descriptions.keys())[:MAX_IMAGES]
    
    X_img_list = []
    X_text_list = []
    y_list = []

    count = 0
    for img_id in image_ids:
        img_path = os.path.join(IMAGE_DIR, img_id)
        
        if not os.path.exists(img_path):
            continue
            
        # Extraction Image (Partie 1)
        # Cela transforme l'image en vecteur (2048,)
        img_feature = vision_part1.extract_image_features(cnn_model, img_path)
        
        # Extraction Labels (Partie 1 - Multi-label)
        # On récupère les tags (chien, chat...) pour savoir quoi prédire
        labels = vision_part1.captions_to_multilabel(descriptions[img_id])
        
        # Si l'image n'a aucun des labels connus, on la saute (optionnel)
        if labels.sum() == 0:
            continue

        # Préparation Texte (Partie 2) 
        # On utilise toutes les descriptions pour multiplier les données d'entraînement
        for description in clean_desc[img_id]:
            # On encode la phrase courante
            text_seq = tokenizer.texts_to_sequences([description])[0]
            text_padded = pad_sequences([text_seq], maxlen=max_length, padding='post')[0]

            # Ajout aux listes (On duplique l'image et le label pour chaque phrase)
            X_img_list.append(img_feature)
            X_text_list.append(text_padded)
            y_list.append(labels)
        
        count += 1
        if count % 200 == 0:
            print(f"Traitement : {count}/{len(image_ids)} images...")

    # Conversion en tableaux Numpy
    X_img = np.array(X_img_list, dtype=np.float32)
    X_text = np.array(X_text_list, dtype=np.int32)
    y = np.array(y_list, dtype=np.float32)

    print(f"Données prêtes : {X_img.shape[0]} exemples d'entraînement.")

    # Séparation Train / Validation
    X_img_train, X_img_val, X_text_train, X_text_val, y_train, y_val = train_test_split(
        X_img, X_text, y, test_size=0.2, random_state=42
    )

    # etape 3 : fusion et entraînement (Partie 3)
    print("\n[4/5] Création du Modèle de Fusion...")
    
    model = fusion_part3.build_fusion_model(
        vocab_size=vocab_size,
        max_length=max_length,
        embedding_dim=300,
        embedding_matrix=embedding_matrix,
        n_classes=len(vision_part1.CLASSES) # On récupère le nombre de classes (12) depuis part1
    )
    
    model.summary()
    
    
    # Sauvegarde uniquement le meilleur modèle
    checkpoint = ModelCheckpoint(
        "modele_fusion.h5",
        monitor="val_accuracy",     # On surveille la précision de validation
        save_best_only=True,        # On écrase le fichier uniquement si on bat le record
        mode="max",                 # On veut le maximum de précision
        verbose=1
    )
    # Arrête l'entraînement si on ne progresse plus (évite de perdre du temps)
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,                 # Si pas d'amélioration pendant 5 époques -> STOP
        restore_best_weights=True
    )
    # Réduit la vitesse d'apprentissage si on stagne (pour affiner le résultat)
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=3,
        min_lr=0.00001
    )

    print("\n[5/5] Lancement de l'entraînement")
    history = model.fit(
        [X_img_train, X_text_train], 
        y_train,
        validation_data=([X_img_val, X_text_val], y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )

    # Pas besoin de model.save() manuel à la fin, car ModelCheckpoint l'a fait pendant l'entraînement
    print("Le meilleur modèle a été sauvegardé sous 'modele_fusion.h5'")

    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Précision du Modèle Multi-Modal')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()