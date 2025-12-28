

**L'entrée du modèle : une image brute (traitée partie 1) et sa description textuelle (traitée partie 2).**

BDD utilisé : Flickr8k

*1. vision_part1.py (partie 1)*

Ce module est responsable du traitement des images. Il utilise un réseau de neurones convolutif pré-entraîné, tel que ResNet50 ou VGG16, pour extraire les caractéristiques visuelles. Chaque image est ainsi transformée en un vecteur numérique.

*3. text_part2.py (partie 2)*

Utilise le fichier glove.6B.200d.txt (pour traduire les mots en vecteur) https://nlp.stanford.edu/data/glove.6B.zip
Ce script gère la description. Il nettoie les descriptions, effectue la tokenisation et utilise un réseau récurrent de type LSTM. Son rôle est de convertir les séquences de mots en vecteurs sémantiques qui capturent le contexte et le sens des phrases.

*5. fusion_part3.py (partie 3)*

Ce fichier définit l'architecture de fusion multi-modale. Il récupère les vecteurs de sortie des modules de vision (partie 1) et de texte (partie 2), les concatène, puis les passe à travers des couches entièrement connectées (Dense) pour produire la classification ou la prédiction finale.

*7. main.py*

Ce script permet le chargement des données du dataset Flickr8k et le processus d'entraînement.
Il exécute la boucle d'apprentissage pour ajuster les poids de l'ensemble du modèle (fine-tuning) et sauvegarde le résultat final dans un fichier .h5.

(Rajotuer un test.py pour tester le fichier .h5 apres que le main.py a été exécuté)
