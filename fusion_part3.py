import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout
from tensorflow.keras.models import Model
import text_part2 

def build_fusion_model(vocab_size, max_length, embedding_dim=300, embedding_matrix=None, n_classes=12):
    """
    Construit le modèle de fusion : 
    Image (2048) + Texte (Sequence) -> Prédiction des labels (Multi-label)
    """
    
    # IMAGE
    # On reçoit le vecteur de 2048 caractéristiques venant de ResNet
    image_input = Input(shape=(2048,), name="image_input")
    x_img = Dense(512, activation="relu")(image_input)
    x_img = Dropout(0.3)(x_img)
    x_img = Dense(256, activation="relu")(x_img)

    # TEXTE
    text_input = Input(shape=(max_length,), name="text_input")
    
    # On utilise ton modèle de la partie 2 comme une sous-couche
    text_branch_model = text_part2.build_text_model(
        vocab_size=vocab_size, 
        max_length=max_length, 
        embedding_dim=embedding_dim, 
        embedding_matrix=embedding_matrix
    )
    
    # On passe l'entrée texte dans ce modèle
    x_text = text_branch_model(text_input)

    # --- FUSION (Concaténation) ---
    # On colle les vecteurs Image (256) et Texte (256) ensemble
    x = Concatenate()([x_img, x_text])
    
    # Couches de décision après fusion
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)

    # Sortie : Classification Multi-label (Sigmoid car plusieurs tags possibles par image)
    output = Dense(n_classes, activation="sigmoid", name="output")(x)

    # Création du modèle final
    model = Model(inputs=[image_input, text_input], outputs=output, name="Fusion_Model")
    
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    
    return model