Ce README est en français. Pour la version anglaise, voir [README_en.md](README_en.md).

# Prédiction du Risque de Diabète — Système Intelligent Biomédical

## Description du Projet

Ce projet vise à développer un système intelligent capable de prédire si un patient présente un **risque élevé de développer le diabète**, à partir de critères cliniques tels que :

- **Glucose** : glycémie  
- **Blood Pressure** : pression artérielle  
- **Skin Thickness** : épaisseur du pli cutané  
- **Insulin** : insuline  
- **BMI** : Indice de Masse Corporelle  
- **Diabetes Pedigree Function** : prédisposition génétique  
- **Age** : âge du patient  

Le système doit permettre :  

1. **Classification supervisée** : identifier les patients à risque élevé ou faible de diabète.  
2. **Clustering non supervisé** : regrouper les patients selon leurs caractéristiques pour identifier des profils similaires au sein de la population.  

Le projet s’appuie sur un jeu de données historiques de patients provenant du laboratoire biomédical.

---

## Technologies Utilisées

- **Python 3** : langage principal pour l'analyse de données et le développement des modèles.  
- **Pandas / NumPy** : manipulation, nettoyage et traitement des données.  
- **Matplotlib / Seaborn** : visualisation des distributions et corrélations.  
- **Scikit-learn** : prétraitement, modèles de classification et clustering, pipelines.  
- **XGBoost** : modèle de gradient boosting pour la classification.   
- **Joblib** : sauvegarde et chargement des modèles.  
- **Streamlit** : interface utilisateur interactive pour saisir les données et visualiser le risque en temps réel.

---

## Structure du projet

```
📁 Analyse-et-Prediction-du-Risque-de-Diabete
│
├── 📄 requirements.txt                 # Dépendances
├── 📄 README.md                        # Documentation du projet en Français
├── 📄 README_en.md                     # Documentation du projet en anglais
├── 📄 main.py                          # Application Streamlit
├── 📁 data/                
├    ├── 📁 raw/                        # Données brutes         
├    └── 📁 processed/                  # Données propres
├── 📁 models/          
│    ├── 📄 model.pkl                   # Modèle final de prédiction du risque de diabète
│    └── 📄 scaler.pkl                  # Scaler utilisé pour normaliser les données
├── 📁 notebooks/
│    ├── 📄 EDA.ipynb                   # Analyse exploratoire des données biomédicales
│    ├── 📄 preprocessing.ipynb         # Prétraitement des variables
│    ├── 📄 Clustering_KMeans.ipynb     # Analyse non supervisée
│    ├── 📄 Classification.ipynb        # Identifier le cluster à haut risque
│    └── 📄 Model_Evaluation.ipynb      # Évaluation et comparaison des performances des modèles
```

---

## Instructions d’Exécution

1. Cloner le projet :  
```bash
git clone https://github.com/anass17/Analyse-et-Prediction-du-Risque-de-Diabete
cd Analyse-et-Prediction-du-Risque-de-Diabete
```

2. Installer les dépendances :
```Bash
pip install -r requirements.txt
```

3. **Lancer l’application Streamlit :**
```bash
streamlit run main.py
```

4. **Ouvrir l’application dans votre navigateur:**
Streamlit ouvrira automatiquement une fenêtre locale, sinon rendez-vous sur : `http://localhost:8501/`

---

## Feature Stories & Tâches

### User Story 1 : Chargement et Analyse Exploratoire des Données (EDA)

- Importer les données avec **Pandas**.  
- Comprendre la structure générale du dataset (types, dimensions, aperçu des valeurs).  
- Identifier les valeurs manquantes et les doublons.  
- Analyser la distribution des variables numériques.  
- Étudier les relations entre variables avec des matrices de corrélation et visualisations graphiques.

---

### User Story 2 : Prétraitement des Données

- Traiter les valeurs manquantes.  
- Détecter les outliers via **boîte à moustaches, z-score, IQR**.  
- Gérer les lignes contenant des valeurs aberrantes.  
- Sélectionner les variables présentant la plus grande variabilité.  
- Visualiser les relations entre variables avec des pairplots.  
- Normaliser ou standardiser les variables numériques.

---

### User Story 3 : Clustering avec K-Means

- Déterminer la valeur optimale de **k** via la méthode du coude et la silhouette.  
- Visualiser la courbe d’inertie et de silhouette.  
- Entraîner le modèle **K-Means** avec le nombre de clusters choisi.  
- Ajouter une colonne `Cluster` indiquant le groupe assigné à chaque patient.  
- Visualiser la répartition des observations par cluster.

---

### User Story 4 : Analyse des Clusters

- Calculer les moyennes des caractéristiques pour chaque cluster.  
- Compter le nombre d’observations par cluster.  
- Identifier les clusters à **haut risque** : par exemple, Glucose >126, BMI >30, Diabetes Pedigree Function >0,5.  
- Ajouter une colonne `risk_category` basée sur le numéro de cluster (1 = risque élevé, 0 = faible).

---

### User Story 5 : Classification Supervisée et Évaluation des Modèles

- Définir la variable cible **y** à partir de la colonne Cluster.  
- Définir les features **X** à partir des variables sélectionnées.  
- Diviser le dataset en **train/test** (80/20).
- Gérer le déséquilibre des classes.
- Tester différents modèles de classification :  
  - Random Forest  
  - SVM  
  - Gradient Boosting  
  - Decision Tree  
  - Régression Logistique  
  - XGBoost  
- Évaluer les modèles avec : matrice de confusion, précision, rappel, F1-score.  
- Validation croisée pour tester la robustesse.  
- Hyperparameter tuning avec **GridSearchCV / RandomizedSearchCV**.  
- Comparer les performances et sélectionner le modèle final.  
- Sauvegarder le modèle final

---

### User Story 6 : Interface Utilisateur

- Développer une application **Streamlit** pour permettre à l’utilisateur de :  
  - Saisir ses données personnelles.  
  - Visualiser en temps réel son risque de diabète.

---

### Évaluation des Modèles

| Model                | Accuracy           |
| ---------------------| ------------------ |
| Random Forest        | 0.9060402684563759 |
| XGB                  | 0.9530201342281879 |
| Gradient Boosting    | 0.9530201342281879 |
| Decision Tree        | 0.8590604026845637 |
| SVM                  | 0.9798657718120806 |
| Logistic Regression  | 0.9865771812080537 |

**Choix final → Logistic Regression**

## Modèle Final

Le modèle final est sauvegardé dans :

```
models/model.pkl
```

Il peut être chargé avec :

```python
import joblib
model = joblib.load("models/model.pkl")
```

---

## Visualisations du projet

### La distribution de données
![Données initiales](https://github.com/user-attachments/assets/a3b30faf-5c5a-4f3a-b41d-32e036b954f7)
![Après le traitement](https://github.com/user-attachments/assets/f740bc4a-d4f7-4f03-b302-c420638d3570)

### Interface Streamlit
![Streamlit UI](https://github.com/user-attachments/assets/2828820c-2b72-4ccf-9046-919f2453c5b7)

---