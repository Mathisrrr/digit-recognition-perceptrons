import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math


#Importation des dataset de test et de train
data,test=pd.read_csv('mnist_train.csv'),pd.read_csv('mnist_test.csv')

y = np.array(data.iloc[:, 0]) #On crée une liste avec tout les nombres
nb_zero =np.sum(y == 0)
x = np.array(data.iloc[:, 1:]) #On crée une liste qui contient les images de tout les nombres
def normal(liste):
    return liste/255

x=normal(x)


class pereceptron:

    def __init__(self,valeur,liste_img,liste_true):
        self.valeur=valeur
        self.bigliste=liste_img  #liste des pixel
        self.liste_true=liste_true  #liste des nombres
        self.weight=np.zeros(28*28)  #liste des poids
        self.weight2=np.zeros(28*28)
        self.c=0 #biais
        self.cp=10  #Poids du biais
        self.seuil= 10 #seuil activation
        self.rate= 1   #taux d'apprentissage

    def prévision(self,liste,listepoids):
        s=self.c*self.cp
        for i in range(len(liste)) :
            s+=liste[i]*listepoids[i]
        return s

    def activate(self,z):
        if z>self.seuil:
            return 1
        else:
            return 0

    def activate2(self,z):                  #----->Moins efficace que la classique cv aussi vers le classique
        if abs(z)>self.seuil:
            return 1 if z>self.seuil else 0
        else:
            return round(z/self.seuil,3)

    def activate_sigmoide(self,z):          #Converge vers la fonction classique (voir avec LR grand)

        return  1/(1+np.exp(-z))

    def activate_tan(self,z):                   #Vraiment moins bien, pas de convergence

        return math.atan(z) if math.atan(z)>0 else 0

    #back propagation
    def maj(self,valeuractivation,liste,label,liste_de_poids):    #liste contient les pixels de l'image en train d'être traitée
        liste_de_poids += self.rate * (label - valeuractivation) * liste
        return liste_de_poids

    def majcarré(self, valeuractivation, liste, label,liste_de_poids):  #Moins bien ou pareil que la première si utlisé avec l'activation classique
        liste_de_poids += self.rate * ((label - valeuractivation)**2) * liste
        return liste_de_poids
    def lancer_le_train(self,boucle=2):
        self.errors_classic = []  # Stocke les erreurs pour la fonction d'activation classique
        self.errors_2 = []  # Stocke les erreurs pour la fonction alternative

        for train in range(boucle):
            error1=0         #On intialise le nombre d'erreur
            error2=0
            for i in range(len(self.bigliste)):
                liste=self.bigliste[i]
                s1=self.prévision(liste,self.weight)
                s2=self.prévision(liste,self.weight2)
                val=self.activate(s1)
                val2=self.activate(s2)
                if self.liste_true[i]==self.valeur:        #Le nombre est celui qu'on veut reconnaitre
                    label=1
                else:                                       #C'est un autre nombre
                    label=0
                if label!=val:                              #Si le résultat n'est pas bon on met à jour les poids
                    error1+=1
                    self.weight = self.maj(val, liste, label,self.weight)
                if label!=val2:
                    error2+=1
                    self.weight2=self.majcarré(val2,liste,label,self.weight2)
            self.rate=self.rate*0.95 if train%2==0 else self.rate
            self.errors_classic.append(error1)
            self.errors_2.append(error2)
            print(f"Nombre d'erreurs sur le passage {train + 1}: {error1} pour le premier et {error2} pour le deuxième")

        self.graph()

    def graph(self):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(launch.errors_classic) + 1), launch.errors_classic, label="Activation Classique (Seuil)",
                 color="red")
        plt.plot(range(1, len(launch.errors_2) + 1), launch.errors_2, label="Activation autre",
                 color="blue")
        plt.xlabel("Itérations")
        plt.ylabel("Nombre d'erreurs")
        plt.title("Comparaison des performances des fonctions d'activation")
        plt.legend()
        plt.grid()
       # plt.show()

    def lancer_le_test(self,attack):
        ltest = np.array(test.iloc[:, 1:])
        if attack==1:#Variation aléatoire fixe sur tout les pixels
           self.modif(ltest)
        if attack == 3:#Saturation de x pixel
            self.modif2(ltest, 200)
        ltest=normal(ltest)
        if attack == 2:#Bruit gaussien
            self.bruit(ltest,pourcentage=0.05)

        ltestnb = np.array(test.iloc[:, 0])
        cpt=0
        for i in range (len(test)):
            liste=ltest[i]

            s=self.prévision(liste,self.weight)
            val=self.activate(s)

            if ltestnb[i]==self.valeur:
                result=1                    #c'est celui qu'on essaye d'identifier
            else:
                result=0                    #Le nombre n'est pas celui qu'on cherche
            if val==result:                   #Si la prédiction est bonne
                cpt+=1
        return cpt/len(ltestnb)


    def modif(self,ltest):
        rng = np.random.default_rng()
        for i in range (len(ltest)):
            ltest[i] = ltest[i] + rng.uniform(-200, 200, 784)
            ltest[i] = np.where(ltest[i] < 0, 0, ltest[i])
            ltest[i] = np.where(ltest[i] > 255, 255, ltest[i])
        return ltest


    def bruit(self,ltest,pourcentage):
        for i in range (len(ltest)):
            ltest[i]=ltest[i]+np.random.normal(0,pourcentage,784)
            ltest[i]=np.where(ltest[i]<0,0,ltest[i])
            ltest[i]=np.where(ltest[i]>1,1,ltest[i])
        return ltest

    def modif2(self,ltest,nbpixel):
        for i in range(len(ltest)):
            indice=np.random.choice(784,nbpixel,replace=False)
            ltest[i][indice]=255
        return ltest

launch=pereceptron(0,x,y)
launch.lancer_le_train()
taux_de_réussite=launch.lancer_le_test(attack=0)
print(taux_de_réussite)

#Affichage des poids finaux
moule=np.array(launch.weight)
moule=moule.reshape(28,28)
plt.imshow(moule,cmap="grey")
plt.show()