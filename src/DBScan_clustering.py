from math import sqrt
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

"""""""""
Definir les deux fonctions qui calculent les distances euclidienne et de manhattan
"""""""""


def distance_manhattan(a, b):
    return sum(abs(valeurA - valeurB) for valeurA, valeurB in zip(a, b))


def distance_euclidienne(a, b):
    if len(a) != len(b):
        return None
    valeur_zip = zip(a, b)
    valeur_difference = [(x[0] - x[1]) ** 2 for x in valeur_zip]
    valeur_somme = sum(valeur_difference)
    distance = sqrt(valeur_somme)
    return distance


"""""""""
Cette fonction retourne SEULEMENT les noeuds admissibles
"""""""""


def Calcul_distance(donnees, noeud_depart, distance_Autour_Noeud, distance):
    # declaration d'une liste des noeuds voisins
    voisins = []

    # Pour chaque noeud de la liste des vecteurs donnees
    for Noeud_Coeur in range(0, len(donnees)):
        # on rajoute les voisins admissibles à la liste des voisins
        if (distance == "euclidienne") and distance_euclidienne(donnees[noeud_depart], donnees[Noeud_Coeur]) < distance_Autour_Noeud:
            voisins.append(Noeud_Coeur)
        elif (distance == "manhattan") and distance_manhattan(donnees[noeud_depart], donnees[Noeud_Coeur]) < distance_Autour_Noeud:
            voisins.append(Noeud_Coeur)
        elif distance not in ["manhattan", "euclidienne"]:
            print("La distance est inconnue")

    return voisins


"""""""""    
La fonction remplir_cluster() permet à travers
les parametres donnees et et noeud de depart de remplir completement cluster

"""""""""


def remplir_cluster(donnees, tags, noeud_depart, noeud_Bordure, Cluster, distance_Autour_Noeud, nbre_noeuds_min,
                    distance):
    tags[noeud_depart] = Cluster
    i = 0
    while i < len(noeud_Bordure):
        Noeud_Coeur = noeud_Bordure[i]
        if tags[Noeud_Coeur] == -1:
            tags[Noeud_Coeur] = Cluster
        elif tags[Noeud_Coeur] == 0:
            tags[Noeud_Coeur] = Cluster
            VoisinP_Test = Calcul_distance(donnees, Noeud_Coeur, distance_Autour_Noeud, distance)
            if len(VoisinP_Test) >= nbre_noeuds_min:
                noeud_Bordure = noeud_Bordure + VoisinP_Test
        i = i + 1


class DBScan_clustering:

    def __init__(self, distance_Autour_Noeud=0.1, nbre_noeuds_min=3):
        self.distance_Autour_Noeud = distance_Autour_Noeud
        self.nbre_noeuds_min = nbre_noeuds_min

    def main(self, donnees, distance):

        distance_Autour_Noeud = self.distance_Autour_Noeud
        Nombre_min = self.nbre_noeuds_min

        # initialiser les tags à ZERO
        tags = [0] * len(donnees)

        Cluster = 0
        for index_point in range(0, len(donnees)):

            if not (tags[index_point] == 0):
                continue

            noeud_Bordure = Calcul_distance(donnees, index_point, distance_Autour_Noeud, distance)

            # si les noeuds bordures sont inferieurs au nbre de noeuds min
            if len(noeud_Bordure) < Nombre_min:
                tags[index_point] = -1

            else:
                Cluster = Cluster + 1
                remplir_cluster(donnees, tags, index_point, noeud_Bordure, Cluster, distance_Autour_Noeud, Nombre_min,
                                distance)
        self.tag = np.array(tags)



###################################################################################################

"""""""""""
Parametres saisis par l'utilisateur à savoir distance autour d'un noeud et nombre minimal de noeuds
"""""""""""
distance_Autour_Noeud = 5
nbre_noeuds_min = 3

###################################################################################################

path_file = os.getcwd()
image_originale = path_file+"/../img/bird.png"
image_selectionnee = cv2.imread(image_originale)



redimentionnee = cv2.resize(image_selectionnee, (100,100), interpolation = cv2.INTER_AREA)
cv2.imwrite(path_file + "/../img/bird_redimentionnee.png",redimentionnee)



# convertion de l'image en RGB
image_convertie = cv2.cvtColor(redimentionnee, cv2.COLOR_BGR2RGB)
troisD = image_convertie.reshape((-1, 3))
troisD = np.float32(troisD)

distanceM = DBScan_clustering(distance_Autour_Noeud=distance_Autour_Noeud, nbre_noeuds_min=nbre_noeuds_min)
distanceM.main(troisD[:, :2], distance="manhattan")
classification = distanceM.tag
First_image_finale = np.uint8(classification.reshape(image_convertie.shape[:2]))


distanceE = DBScan_clustering(distance_Autour_Noeud=distance_Autour_Noeud, nbre_noeuds_min=nbre_noeuds_min)
distanceE.main(troisD[:, :2], distance="euclidienne")
classification2 = distanceE.tag
Second_image_finale = np.uint8(classification2.reshape(image_convertie.shape[:2]))


taille_figure = 15
plt.figure(figsize=(taille_figure, taille_figure))

"""""""""""
Affichage photo originale
"""""""""""
plt.subplot(2, 2, 1), plt.imshow(image_convertie)
plt.title('Photo originale'), plt.xticks([]), plt.yticks([])

"""""""""""
Affichage photo with DBScan + distance manhattan
"""""""""""

plt.subplot(2, 2, 2), plt.imshow(First_image_finale)
plt.title('DBScan + distance manhattan (distance_Autour_Noeud = %0.2f)' % distance_Autour_Noeud), plt.xticks([]), plt.yticks([])



"""""""""""
Affichage photo with DBScan + distance euclidienne
"""""""""""

plt.subplot(2, 2, 3), plt.imshow(Second_image_finale)
plt.title('DBScan + distance euclidienne (distance_Autour_Noeud = %0.2f)' % distance_Autour_Noeud), plt.xticks([]), plt.yticks([])

plt.savefig(path_file+'/../output/DBScan_%0.2f.png'%distance_Autour_Noeud)
plt.show()