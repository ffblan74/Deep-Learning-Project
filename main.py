import os
import numpy as np
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.sequence import pad_sequences


import vision_part1
import text_part2
import fusion_part3

# Config globale
IMAGE_DIR = "Flicker8k_Dataset"
TOKEN_FILE = "Flickr8k.token.txt"
GLOVE_FILE = "glove.6B.300d.txt"
MAX_IMAGES = 2000   # Réduit pour éviter de saturer la RAM avec le Fine-Tuning
EPOCHS = 5         

def main():
    print("PROJET DEEP LEARNING : FINE-TUNING & CROSS-VALIDATION\n")

    #1 texte
    print("[1/4] Préparation du Texte...")
    descriptions = vision_part1.load_flickr_descriptions(TOKEN_FILE)
    if not descriptions: return

    clean_desc = text_part2.clean_descriptions(descriptions)
    all_sentences = [desc for desc_list in clean_desc.values() for desc in desc_list]
    tokenizer = text_part2.create_tokenizer(all_sentences)
    vocab_size = len(tokenizer.word_index) + 1
    max_length = max(len(d.split()) for d in all_sentences)
    
    print(f"Vocabulaire: {vocab_size}, Max Length: {max_length}")

    print("[2/4] Chargement GloVe...")
    embedding_matrix = None
    if os.path.exists(GLOVE_FILE):
        embedding_matrix = text_part2.load_glove_embeddings(GLOVE_FILE, tokenizer.word_index, embedding_dim=300)

    #2 Image (Chargement en RAM pour Fine-Tuning) ---
    print(f"\n[3/4] Chargement des images brutes ({MAX_IMAGES})...")
    # Pour le Fine-Tuning, on ne pré-calcule PAS les features. On charge les pixels.
    
    image_ids = list(descriptions.keys())[:MAX_IMAGES]
    X_img_list, X_text_list, y_list = [], [], []

    count = 0
    for img_id in image_ids:
        img_path = os.path.join(IMAGE_DIR, img_id)
        if not os.path.exists(img_path): continue
            
        labels = vision_part1.captions_to_multilabel(descriptions[img_id])
        if labels.sum() == 0: continue

        # chargement image brute (224, 224, 3)
        # vision_part1.load_and_preprocess_image renvoie (1, 224, 224, 3)
        # on prend [0] pour avoir (224, 224, 3)
        img_pixels = vision_part1.load_and_preprocess_image(img_path)[0]
        
        # On duplique pour chaque description (Data Augmentation textuelle)
        # Pour aller vite, on en prend juste une ou deux ici
        for description in clean_desc[img_id][:2]: 
            text_seq = tokenizer.texts_to_sequences([description])[0]
            text_padded = pad_sequences([text_seq], maxlen=max_length, padding='post')[0]

            X_img_list.append(img_pixels)
            X_text_list.append(text_padded)
            y_list.append(labels)
        
        count += 1
        if count % 100 == 0: print(f"Chargé {count} images")

    X_img = np.array(X_img_list, dtype=np.float32)
    X_text = np.array(X_text_list, dtype=np.int32)
    y = np.array(y_list, dtype=np.float32)
    
    print(f"Dataset pret : {len(X_img)} échantillons.")

    #3 Crossvalidation
    print("\n[4/4] Recherche de parmatetre et Cross-Validation")
    
    # Paramétre a tester Dropout
    dropouts_to_test = [0.3, 0.5]
    
    best_score = 0
    best_dropout = 0
    
    # Boucle d'Optimisation
    for drop_rate in dropouts_to_test:
        print(f"\n test hyperparmatére : Dropout = {drop_rate}")
        
        #Cross-valisation (K-Fold 3 ou 5 splits)
        kfold = KFold(n_splits=3, shuffle=True, random_state=42)
        scores_fold = []
        
        fold_no = 1
        for train, val in kfold.split(X_img):
            print(f"Fold {fold_no}/3 ")
            
            # Création du modèle (Fine-Tuning)
            model = fusion_part3.build_fusion_model_finetuning(
                vocab_size, max_length, embedding_matrix, 
                n_classes=12, dropout_rate=drop_rate
            )
            
            # Callbacks
            early = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
            
            # Entrainement
            model.fit(
                [X_img[train], X_text[train]], y[train],
                validation_data=([X_img[val], X_text[val]], y[val]),
                epochs=EPOCHS, batch_size=32, verbose=0,
                callbacks=[early]
            )
            
            # Évaluation
            loss, acc = model.evaluate([X_img[val], X_text[val]], y[val], verbose=0)
            scores_fold.append(acc)
            print(f"Acc: {acc:.2%}")
            fold_no += 1
            
        avg_score = np.mean(scores_fold)
        print(f"moyenne Dropout {drop_rate} : {avg_score:.2%}")
        
        if avg_score > best_score:
            best_score = avg_score
            best_dropout = drop_rate

    print(f"Resultat final")
    print(f"Meilleur Dropout : {best_dropout} avec une précision de {best_score:.2%}")
    print("Modèle validé par Cross-Validation et Fine-Tuning.")

    # Meilleur modèle final
    final_model = fusion_part3.build_fusion_model_finetuning(
        vocab_size, max_length, embedding_matrix, 
        n_classes=12, dropout_rate=best_dropout
    )
    
    # On met un peu plus d'époques pour le modèle final
    final_model.fit(
        [X_img, X_text], y,
        epochs=EPOCHS, 
        batch_size=32,
        verbose=1
    )
    final_model.save("modele_fusion.h5")
    print("\nModèle sauvegardé sous 'modele_fusion.h5'")
    print("Prêt pour le test.py")

if __name__ == "__main__":
    main()