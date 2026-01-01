import os
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt

#etape 1
def load_flickr_descriptions(filename):
    """
    Reads the Flickr8k token file and returns a dictionary:
    image_id -> list of descriptions
    """
    mapping = {}
    
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            text = file.read()
            
        for line in text.split('\n'):
            # Skip empty lines
            if len(line) < 2:
                continue
                
            tokens = line.split()
            
            # First token is: image.jpg#0
            image_id = tokens[0]
            
            # Rest is the caption
            image_desc = ' '.join(tokens[1:])
            
            # Remove #0, #1...
            image_id = image_id.split('#')[0]
            
            if image_id not in mapping:
                mapping[image_id] = []
            
            mapping[image_id].append(image_desc)
            
        print(f"Loaded {len(mapping)} images from {filename}")
        return mapping

    except FileNotFoundError:
        print(f"ERROR: File '{filename}' not found.")
        return {}

#etape 2
CLASSES = ["person", "child", "dog", "cat", "horse", "car", "bike", "ball",
           "water", "food", "street", "nature"]

KEYWORDS = { "person": ["man", "woman", "person", "people", "men", "women"],
            "child":  ["child", "children", "boy", "girl", "kid"],
            "dog":    ["dog", "puppy"],
            "cat":    ["cat", "kitten"],
            "horse":  ["horse"],
            "car":    ["car", "truck", "bus"],
            "bike":   ["bike", "bicycle", "cyclist"],
            "ball":   ["ball"],
            "water":  ["water", "ocean", "sea", "river", "lake", "beach", "pool", "surf"],
            "food":   ["food", "pizza", "sandwich", "cake", "bread", "meal"],
            "street": ["street", "road", "city", "urban"],
            "nature": ["tree", "trees", "forest", "mountain", "field", "grass"]}

def captions_to_multilabel(desc_list):
    """
    Convertit les captions d'une image en vecteur multi-label.
    Sortie : array de taille len(CLASSES) avec des 0 et des 1.
    """
    text = " ".join(desc_list).lower()
    y = np.zeros(len(CLASSES), dtype=np.float32)
    for i, cname in enumerate(CLASSES):
        for kw in KEYWORDS[cname]:
            if re.search(rf"\b{re.escape(kw)}\b", text):
                y[i] = 1.0
                break

    return y

#code test
"""
if __name__ == "__main__":
    filename = "Flickr8k_text/Flickr8k.token.txt"
    #Charge les descriptions
    descriptions = load_flickr_descriptions(filename)
    if len(descriptions) == 0:
        print("Aucune description chargée.")
        exit()
    #Prend une image au hasard
    first_img = list(descriptions.keys())[0]
    # 3. Convertit ses captions en labels
    y = captions_to_multilabel(descriptions[first_img])
    print("Image :", first_img)
    print("Labels détectés :")
    for i, cname in enumerate(CLASSES):
        if y[i] == 1:
            print("-", cname)
"""


#etape 3
def build_cnn_encoder():
    """
    Charge ResNet50 pré-entraîné (ImageNet)
    et enlève la couche de classification.
    """
    model = ResNet50(
        weights="imagenet",
        include_top=False,
        pooling="avg"  
    )
    return model



def load_and_preprocess_image(image_path):
    """
    Charge une image et la prépare pour ResNet50.
    """
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


def extract_image_features(model, image_path):
    """
    Image → vecteur de caractéristiques (2048)
    """
    img = load_and_preprocess_image(image_path)
    features = model.predict(img, verbose=0)
    return features[0]


#code test
"""
if __name__ == "__main__":
    image_dir = "Flicker8k_Dataset"
    filename = "Flickr8k_text/Flickr8k.token.txt"
    descriptions = load_flickr_descriptions(filename)
    first_img = list(descriptions.keys())[0]
    image_path = os.path.join(image_dir, first_img)
    cnn = build_cnn_encoder()
    vec = extract_image_features(cnn, image_path)
    print("Image :", first_img)
    print("Vecteur image shape :", vec.shape)
"""
#étape 4

def build_dataset(descriptions, image_dir, cnn_model, max_images=2000):
    X_list, y_list = [], []

    image_ids = list(descriptions.keys())[:max_images]

    missing_img = 0
    empty_label = 0
    kept = 0

    for i, img_id in enumerate(image_ids):
        img_path = os.path.join(image_dir, img_id)

        # 1) image existe ?
        if not os.path.exists(img_path):
            missing_img += 1
            continue

        # 2) labels depuis captions
        y = captions_to_multilabel(descriptions[img_id])

        # (temporairement) on compte ceux qui n'ont pas de label
        if y.sum() == 0:
            empty_label += 1
            continue

        # 3) features image
        x = extract_image_features(cnn_model, img_path)

        X_list.append(x)
        y_list.append(y)
        kept += 1

        if (i + 1) % 200 == 0:
            print(f"Processed {i+1}/{len(image_ids)} | kept={kept} | missing={missing_img} | empty_label={empty_label}")

    print("---- DATASET STATS ----")
    print("missing images:", missing_img)
    print("empty labels:", empty_label)
    print("kept samples:", kept)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y


#code test
"""
if __name__ == "__main__":
    image_dir = "Flicker8k_Dataset"
    token_file = "Flickr8k_text/Flickr8k.token.txt"

    descriptions = load_flickr_descriptions(token_file)
    cnn = build_cnn_encoder()

    X, y = build_dataset(descriptions, image_dir, cnn, max_images=500)

    print("X shape =", X.shape)  # attendu: (N, 2048)
    print("y shape =", y.shape)  # attendu: (N, 12)
    print("Exemple y[0] =", y[0], " / nb labels =", y[0].sum())
"""
#etape 5
def build_classifier(input_dim=2048, n_classes=12):
    model = Sequential([
        Dense(512, activation="relu", input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(256, activation="relu"),
        Dropout(0.3),
        Dense(n_classes, activation="sigmoid")  # multi-label
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

#code test
"""
if __name__ == "__main__":

    token_file = "Flickr8k_text/Flickr8k.token.txt"
    image_dir  = "Flicker8k_Dataset"  
    descriptions = load_flickr_descriptions(token_file)
    if len(descriptions) == 0:
        print("Aucune description chargée.")
        exit()
    cnn = build_cnn_encoder()
    X, Y = build_dataset(descriptions, image_dir, cnn, max_images=500)
    print("X shape =", X.shape)
    print("Y shape =", Y.shape)
    if len(Y) == 0:
        print("Dataset vide -> problème de chemin ou de filtrage.")
        exit()
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    model = Sequential([
        Dense(512, activation="relu", input_shape=(X.shape[1],)),
        Dropout(0.3),
        Dense(256, activation="relu"),
        Dropout(0.3),
        Dense(len(CLASSES), activation="sigmoid")
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=5,
        batch_size=32
    )
    pred = model.predict(X_val[:1], verbose=0)[0]
    print("\n--- PREDICTION EXEMPLE ---")
    print("Probabilités:", pred)
    predicted_classes = [CLASSES[i] for i, p in enumerate(pred) if p >= 0.5]
    print("Classes prédites (>=0.5):", predicted_classes)
    true_classes = [CLASSES[i] for i, v in enumerate(y_val[0]) if v == 1]
    print("Vrai labels:", true_classes)
"""

#étape 6
def show_image_prediction(image_path, cnn, model, true_labels=None, threshold=0.5):
    x = extract_image_features(cnn, image_path)
    x = np.expand_dims(x, axis=0)
    pred = model.predict(x, verbose=0)[0]
    predicted_classes = [CLASSES[i] for i, p in enumerate(pred) if p >= threshold]
    top_scores = [(CLASSES[i], float(pred[i])) for i in np.argsort(pred)[::-1][:5]]
    img = load_img(image_path)
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis("off")
    title = f"Predicted: {predicted_classes}"
    if true_labels is not None:
        title += f"\nTrue: {true_labels}"
    plt.title(title)
    plt.show()
    print("Top predictions:")
    for cls, score in top_scores:
        print(f"{cls:10s} → {score:.2f}")


#code test
"""
if __name__ == "__main__":
    token_file = "Flickr8k_text/Flickr8k.token.txt"
    image_dir  = "Flicker8k_Dataset"
    descriptions = load_flickr_descriptions(token_file)
    cnn = build_cnn_encoder()
    X, Y = build_dataset(descriptions, image_dir, cnn, max_images=500)
    n = len(X)
    idx = np.random.permutation(n)
    split = int(0.8 * n)
    X_train, Y_train = X[idx[:split]], Y[idx[:split]]
    X_val,   Y_val   = X[idx[split:]], Y[idx[split:]]
    classifier = build_classifier(input_dim=2048, n_classes=len(CLASSES))
    classifier.fit(X_train, Y_train, epochs=5, batch_size=32, verbose=1)
    img_id = list(descriptions.keys())[idx[split]]
    image_path = os.path.join(image_dir, img_id)
    true_y = captions_to_multilabel(descriptions[img_id])
    true_classes = [CLASSES[i] for i, v in enumerate(true_y) if v == 1]
    show_image_prediction(
        image_path=image_path,
        cnn=cnn,
        model=classifier,
        true_labels=true_classes,
        threshold=0.5
    )
"""