# 1 Préparer les entrées (images + texte + labels)

import numpy as np
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout
from tensorflow.keras.models import Model
import random
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import vision_part1
import text_part2

image_dir = "Flicker8k_Dataset"
token_file = "Flickr8k_text/Flickr8k.token.txt"
glove_path = "glove.6B.300d.txt"

descriptions = vision_part1.load_flickr_descriptions(token_file)
clean_desc = text_part2.clean_descriptions(descriptions)
all_sentences = [desc for desc_list in clean_desc.values() for desc in desc_list]

tokenizer = text_part2.create_tokenizer(all_sentences)
vocab_size = len(tokenizer.word_index) + 1
print(f"Vocabulary size: {vocab_size}")

max_length = max(len(desc.split()) for desc in all_sentences)
print(f"Max sentence length: {max_length}")

if os.path.exists(glove_path):
    embedding_matrix = text_part2.load_glove_embeddings(glove_path, tokenizer.word_index, embedding_dim=300)
else:
    embedding_matrix = None
    print("GloVe non trouvé, embeddings seront entraînés depuis zéro.")

text_model = text_part2.build_text_model(vocab_size, max_length, embedding_dim=300, embedding_matrix=embedding_matrix)

cnn_model = vision_part1.build_cnn_encoder()

max_images = 200  # ou 200 pour des tests rapides
image_ids = list(descriptions.keys())[:max_images]
X_images = []
X_texts = []
y_labels = []

for img_id in image_ids:
    img_path = os.path.join(image_dir, img_id)
    if not os.path.exists(img_path):
        continue

    img_vec = vision_part1.extract_image_features(cnn_model, img_path)
    X_images.append(img_vec)

    seq_list = tokenizer.texts_to_sequences(clean_desc[img_id])
    seq_list = pad_sequences(seq_list, maxlen=max_length, padding='post')
    X_texts.append(seq_list[0])

    y = vision_part1.captions_to_multilabel(descriptions[img_id])
    y_labels.append(y)

X_images = np.array(X_images, dtype=np.float32)
X_texts  = np.array(X_texts, dtype=np.int32)
y_labels = np.array(y_labels, dtype=np.float32)

print("X_images.shape:", X_images.shape)
print("X_texts.shape:", X_texts.shape)
print("y_labels.shape:", y_labels.shape)

# étape 2 Fusion des vecteurs

descriptions = text_part2.load_flickr_descriptions('Flickr8k.token.txt')
clean_desc = text_part2.clean_descriptions(descriptions)
all_sentences = [desc for desc_list in clean_desc.values() for desc in desc_list]
my_tokenizer = text_part2.create_tokenizer(all_sentences)
vocab_size = len(my_tokenizer.word_index) + 1
max_length = max(len(d.split()) for d in all_sentences)
glove_path = 'glove.6B.300d.txt'
embedding_matrix = text_part2.load_glove_embeddings(glove_path, my_tokenizer.word_index, embedding_dim=300)

image_input = Input(shape=(2048,), name="image_input")
text_input  = Input(shape=(max_length,), name="text_input")

x_img = Dense(512, activation="relu")(image_input)
x_img = Dropout(0.3)(x_img)
x_img = Dense(256, activation="relu")(x_img)

x_text = text_part2.build_text_model(
    vocab_size=vocab_size,
    max_length=max_length,
    embedding_dim=300,
    embedding_matrix=embedding_matrix)(text_input)

x = Concatenate()([x_img, x_text])
x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)

n_classes = len(vision_part1.CLASSES)
output = Dense(n_classes, activation="sigmoid", name="output")(x)

fusion_model = Model(inputs=[image_input, text_input], outputs=output)
fusion_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

fusion_model.summary()

# étape 3 Test / prédiction sur de nouvelles images

idx = random.randint(0, len(X_images)-1)
img_id = image_ids[idx]
img_path = os.path.join(image_dir, img_id)

plt.imshow(load_img(img_path))
plt.axis('off')
plt.title(f"Image: {img_id}")
plt.show()

img_vec = vision_part1.extract_image_features(cnn_model, img_path)
img_vec = img_vec.reshape(1, -1)

seq = tokenizer.texts_to_sequences([clean_desc[img_id][0]])
seq = pad_sequences(seq, maxlen=max_length, padding='post')

y_pred = fusion_model.predict([img_vec, seq], verbose=0)

classes = vision_part1.CLASSES
pred_labels = [classes[i] for i, p in enumerate(y_pred[0]) if p >= 0.5]

print("\n--- Résultat prédiction ---")
print("Image :", img_id)
print("Classes prédites (p>=0.5) :", pred_labels)
print("Probabilités :", y_pred[0])

true_labels = vision_part1.captions_to_multilabel(descriptions[img_id])
true_labels_names = [classes[i] for i, val in enumerate(true_labels) if val==1]

print("Labels réels :", true_labels_names)