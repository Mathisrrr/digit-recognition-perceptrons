import numpy as np
from extract6 import ImageExtractor

class MLP:
    def __init__(self, input_size, output_size):
        self.nbneuron = [250,130]
        self.input_size = input_size
        self.output_size = output_size
        self.learningrate = 0.01
        self.best_weights = None
        self.best_acc = 0
        self.launch()

    def launch(self):
        self.listematrixweight = []
        self.matrixbias = []
        layer_sizes = [self.input_size] + self.nbneuron + [self.output_size]

        for i in range(len(layer_sizes) - 1):
            #Initialisation des poids et des biais
            scale = np.sqrt(2.0 / layer_sizes[i])
            self.listematrixweight.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * scale)
            self.matrixbias.append(np.zeros((1, layer_sizes[i + 1])))

    def softmax(self, x):

        x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / (exp_x.sum(axis=1, keepdims=True) + 1e-12)

    def relu(self,x):
        return np.maximum(0,x)

    def sigmoid(self, x):
        z = np.clip(x, -10, 10)
        return 1 / (1 + np.exp(-z))

    def sigmoide_derivative(self,x):
        s=self.sigmoid(x)
        return s*(1-s)

    def relu_derivative(self,x):
        return np.where(x > 0, 1.0, 0)
    def forward(self, X):
        self.activations = [X]
        self.zs = []


        for i in range(len(self.listematrixweight)):
            z = self.activations[-1] @ self.listematrixweight[i] + self.matrixbias[i]

            if i == len(self.listematrixweight) - 1:#rien pour la dernière
                a = z
            else:#Fonction non linéaire
                a = self.relu(z)

            self.zs.append(z)
            self.activations.append(a)
        return self.softmax(self.activations[-1])



    def backward(self, X, y_true):
        m = X.shape[0]
        output = self.forward(X)

        # Gradient initial avec contrôle
        delta = np.clip(output - y_true, -5.0, 5.0) / m

        #On parcourt le réseau à l'envers
        for i in reversed(range(len(self.listematrixweight))):
            #gradient
            grad_w = self.activations[i].T @ delta
            grad_w = np.clip(grad_w, -1.0, 1.0)

            grad_b = np.clip(np.sum(delta, axis=0, keepdims=True), -1.0, 1.0)

            # Mise à jour
            self.listematrixweight[i] -= self.learningrate * grad_w
            self.matrixbias[i] -= self.learningrate * grad_b

            # Propagation pour couches précédentes
            if i > 0:
                delta = delta @ self.listematrixweight[i].T
                delta = np.clip(delta, -5.0, 5.0)

                delta *= self.relu_derivative(self.activations[i])


    def train(self, X, y, epochs=20, batch_size=120):

        y_onehot = np.eye(self.output_size)[y]

        for epoch in range(epochs):
            # Mélange des données
            indices = np.random.permutation(len(X))

            for i in range(0, len(X), batch_size):
                batch_idx = indices[i:i + batch_size]
                self.backward(X[batch_idx], y_onehot[batch_idx])

            # Validation
            preds = np.argmax(self.forward(X), axis=1)
            acc = np.mean(preds == y)

            # Sauvegarde des meilleurs poids
            if acc > self.best_acc:
                self.best_acc = acc
                self.best_weights = [w.copy() for w in self.listematrixweight]
                self.best_biases = [b.copy() for b in self.matrixbias]

            if acc>0.8:
                break


            print(f"Epoch {epoch + 1}/{epochs} - Accuracy: {acc * 100:.2f}% (Best: {self.best_acc * 100:.2f}%)")

        self.listematrixweight=self.best_weights
        self.matrixbias=self.best_biases

    def phrase(self):#Deine des sequences
        guess=input("Entrez la phrase")

        liste_x=[]
        liste_y=[]

        for i in guess:
            z=ord(i)-33

            a=np.where(y_train == z)[0][0]

            liste_x.append(x_train[a])
            liste_y.append(z)

        self.devinerphrase(np.array(liste_x),np.array(liste_y))

    def devinerphrase(self,x,sentence):#Affiche la phrase correctement
        retour=self.test(x,sentence)
        retour+=33
        mot=""
        for i in retour:
            mot+=chr(i)

        print(f"prédiction du modèle: {mot}")


    def test(self, X, y):#Test d'un dataset

        X=X.reshape(len(X),-1)

        preds = np.argmax(self.forward(X), axis=1)

        precision = np.mean(preds == y)  # sort le nombre de GA sur le nombre de try

        print(f'précision du test:taux de réussite: {precision * 100:.2f} %')
        return preds







# Usage
base = ImageExtractor()
base.reconstruct_archive()
x_train, y_train, x_test, y_test = base.sortie()
y_train=y_train.astype(int) - 33
y_test=y_test.astype(int) - 33
# Normalisation
x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
x_test = x_test.reshape(x_test.shape[0], -1) / 255.0


model = MLP(1024, 94)
model.train(x_train, y_train)
model.test(x_test,y_test)
for i in range(3):
    print("vous allez pouvoir écrire 3 mots")
    model.phrase()

print("\nPour ce qui est des plans de test, "
      "\nun learning rate égal à 0,01 est optimal afin de pas faire exploser les valeurs, si on met plus l'apprentissage n'est pas stable"
      "\nNous avons choisi de garder uniquement deux couches afin d'optimer la vitesse de train en remarquant que les performances ne s'amélioraient pas significativement"
      "\nLe nombre de neuronnes par couche à été chosis après une batterie de test"
      "\nLa fonction d'activation choisi est la relu(70%) car plus efficace que la sigmoide(45%)"
      "\nFinalement on arrive à avoir des performance entre 60 et 70% de réussite pour la reconnaissance des 93 caractères")