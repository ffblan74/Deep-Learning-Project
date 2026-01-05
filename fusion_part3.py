import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
import text_part2
import vision_part1

def build_fusion_model_finetuning(vocab_size, max_length, embedding_matrix=None, n_classes=12, dropout_rate=0.3):
    """
    Modèle End-to-End : Image Brute + Texte -> Prédiction
    Intègre le Fine-Tuning du CNN.
    """
    
    # Input des images(Entrée = Pixels) on prend une image couleur 224x224
    image_input = Input(shape=(224, 224, 3), name="image_input")
    
    # On récupère le ResNet de la partie 1
    # Note: pooling="avg" dans vision_part1 fait que la sortie est déjà un vecteur
    cnn_base = vision_part1.build_cnn_encoder()
    
    # fine-tuning On débloque les dernières couches
    cnn_base.trainable = True
    # On gèle les couches du début (les 140 premières sur a peu pres 175) pour ne pas changer les features de base
    for layer in cnn_base.layers[:-30]:
        layer.trainable = False
        
    # On passe l'image dans le CNN
    x_img = cnn_base(image_input)
    
    x_img = Dense(512, activation="relu")(x_img)
    x_img = Dropout(dropout_rate)(x_img)

    # Input du texte
    text_input = Input(shape=(max_length,), name="text_input")
    
    text_branch = text_part2.build_text_model(
        vocab_size=vocab_size, 
        max_length=max_length, 
        embedding_matrix=embedding_matrix
    )
    x_text = text_branch(text_input)

    # Fusion des deux
    x = Concatenate()([x_img, x_text])
    
    x = Dense(256, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(128, activation="relu")(x)
    
    # Sortie Multi-Label
    output = Dense(n_classes, activation="sigmoid", name="output")(x)

    model = Model(inputs=[image_input, text_input], outputs=output, name="Fusion_FineTuning")
    
    # Learning Rate très bas (1e-5) pour le Fine-Tuning
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    
    return model