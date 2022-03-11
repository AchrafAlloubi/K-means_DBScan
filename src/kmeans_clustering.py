from matplotlib import pyplot as plt
import os
import numpy as np
from math import sqrt
import cv2


"""""""""
Definir les deux fonctions qui calculent les distances euclidienne et de manhattan
"""""""""
def distance_manhattan(a, b):
    return sum(abs(valeurA-valeurB) for valeurA, valeurB in zip(a, b))

def distance_euclidienne(a, b):
    if len(a) != len(b):
        return None
    valeur_zip = zip(a, b)
    valeur_difference = [(x[0] - x[1])**2 for x in valeur_zip]
    valeur_somme = sum(valeur_difference)
    distance = sqrt(valeur_somme)
    return distance

def normalisation(a, b):
    return np.linalg.norm(a-b)

class kmeans_clustering:

    def __init__(self, k, nbre_repetition):
        self.k = k
        self.nbre_repetition = nbre_repetition

    def main(self, donnees, distance_choisie):
        self.centres = {}
        self.tags = []

        for i in range(self.k):
            self.centres[i] = donnees[i]  #Initialisation des centres

        for i in range(self.nbre_repetition):
            self.categories = {}          #Contient les centres et les classifications

            for i in range(self.k):       #Les clés deviennent centres et les valeurs -> entite qui sont dans les valeurs
                self.categories[i] = []
            for entite in donnees:
                distance = [normalisation(entite, self.centres[barycentre]) for barycentre in self.centres]

                #retourne la position de la premiere occurence de distance minimale
                clustering = distance.index(min(distance))
                self.categories[clustering].append(entite)

            for clustering in self.categories:
                if self.categories[clustering] != []:
                    self.centres[clustering] = np.average(self.categories[clustering], axis=0)

        for entite in donnees:
            if distance_choisie == "euclidienne":
                distance = [distance_euclidienne(entite, self.centres[barycentre]) for barycentre in self.centres]
            elif distance_choisie == "manhattan":
                distance = [distance_manhattan(entite, self.centres[barycentre]) for barycentre in self.centres]

            clustering = distance.index(min(distance))
            self.tags.append(clustering)

###################################################################################################

"""""""""""
Parametres saisis par l'utilisateur à savoir distance autour d'un noeud et nombre minimal de noeuds
"""""""""""

K = 16
repetition = 10

###################################################################################################


path_file = os.getcwd()
image_originale = path_file+"/../img/bird.png"

image_selectionnee = cv2.imread(image_originale) #imageSelectionnee


# convertion de l'image en RGB
image_convertie = cv2.cvtColor(image_selectionnee,cv2.COLOR_BGR2RGB)
troisD = image_convertie.reshape((-1, 3))
troisD = np.float32(troisD)


distanceE = kmeans_clustering(k=K, nbre_repetition=repetition)
distanceE.main(troisD, distance_choisie="euclidienne")
center = np.uint8(list(distanceE.centres.values()))
resultat = center[distanceE.tags]
First_image_finale = resultat.reshape((image_convertie.shape))

distanceM = kmeans_clustering(k=K,nbre_repetition=repetition)
distanceM.main(troisD, distance_choisie="manhattan")
center2 = np.uint8(list(distanceM.centres.values()))
Second_resultat = center2[distanceM.tags]
Second_image_finale = Second_resultat.reshape((image_convertie.shape))


taille_figure = 15
plt.figure(figsize=(taille_figure, taille_figure))

#affichage de l'image originale
plt.subplot(2,2,1), plt.imshow(image_convertie)
plt.title('Original'), plt.xticks([]), plt.yticks([])

#affichage de l'image segmentee + distance euclidienne
plt.subplot(2,2,2),plt.imshow(First_image_finale)
plt.title('K = %i distance euclidienne ' % K), plt.xticks([]), plt.yticks([])


#affichage de l'image segmentee + distance manhattan
plt.subplot(2,2,4),plt.imshow(Second_image_finale)
plt.title(' K = %i  distance manhattan' % K), plt.xticks([]), plt.yticks([])
plt.savefig(path_file+'/../output/Kmeans_%i.jpg'%K)
plt.show()