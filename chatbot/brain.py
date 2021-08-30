#!/usr/bin/env python3

# brain version 0.1
# Copyright (C) 2021 Alexandre Martos -- alexandre.martos (at) protonmail.ch
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

# *****************************************************************************
# Nom     : brain
# Rôle    : module gérant un modèle de prédiction de réponses d'un chatbot. Le
#           modèle est entraîné via la commande "train" et un fichier décrivant
#           les différentes intentions (cf le module "patterns"), et peut
#           prédire les réponse à une phrase à l'aide de la commande "answer"). 
#           Ce bot est inspiré de :
#               - https://github.com/jerrytigerxu/Simple-Python-Chatbot
#               - https://databotlab.com/creating-a-neural-network-chatbot-using-tensorflow-and-nltk/
# Auteur  : Martos Alexandre
# E-mail  : alexandre.martos@protonmail.ch
# Version : 0.1
# Licence : GPLv3
# Python  : Python 3.9.2
# Keras   : 2.3.1
# Usage   :
#     créer un modèle     : ChatBotBrain().train("mymodel", "myintents.json")
#     prédire une réponse :
#                           model = ChatBotBrain("mymodel.h5", "myintents.pkl")
#                           answer = model.answer("Hello World !")
# *****************************************************************************

import os
import numpy

# keras - une librairie gérant gérant l'IA basée sur Theano.
# https://fr.wikipedia.org/wiki/Theano_(logiciel)
# contextlib est une librairie standard de gestion de contextes
import contextlib
import sys
# Keras affiche un message sur stderr dès l'import. On le redirige vers
# /dev/null
with contextlib.redirect_stderr(open(os.devnull, 'w')):
    from keras.models import Sequential, load_model
    from keras.layers import Dense, Activation, Dropout
    from keras.optimizers import SGD

# la librairie traitant les intentions
from . import Patterns

class ChatBotBrain:
    """
    L'IA d'un chatbot, permettant de prédire une réponse à partir d'une phrase.

    Utilise le package patterns pour convertir les intentions dans un format
    approprié. Ce package stocke les données dans un fichier pickle. Si
    l'instance de ChatBotBrain est chargée avec un modèle pré-construit, c'est
    ce fichier pickle qui représente la liste des intentions.

    Methods
    -------
    train(model_name, intents_path)
        Créé le modèle de l'IA à partir d'un fichier d'intentions.

    answer(sentence)
        Prédit l'intention de la phrase donnée et renvoie une réponse.

    Examples
    --------
    >>> model = ChatBotBrain() # verbose=False pour masquer le rapport de fit
    >>> model.train("mymodel", "myintents.json")
    >>> model.answer("Hello World !")
    'Hi !'
    >>> other_instance = ChatBotBrain("mymodel.h5", "myintents.pkl")
    >>> other_instance.answer("Hello World !")
    'Hi !'

    See also
    --------
    patterns : module de pré-traitement des intentions.
    """

    def __init__(self, model_path="", pickle_path="", verbose=True):
        """
        Initialise la classe ChatBotBrain.

        Parameters
        ----------
        model_path : string
            Le chemin d'un fichier h5, un modèle de prédiction pré-construit.
        patterns_path : string
            Le chemin d'un fichier pickle, stockant les intentions pré-traitées
            par patterns.
        verbose : bool
            Un booléen indiquant si l'on souhaite un rapport détaillé du fitting
            du modèle.
        """
        if model_path or pickle_path:
            for f in [model_path, pickle_path]:
                if not os.path.isfile(f):
                    raise FileNotFoundError(f)
        self._verbose = verbose
        self._patterns = Patterns(pickle_path)
        self._model = load_model(model_path) if model_path else Sequential()

    def train(self, model_name, intents_path):
        """
        Construit un modèle de prédiction à partir d'un fichier d'intentions.

        Parameters
        ----------
        model_name : string
            Le nom du modèle, servant de base du nom de fichier h5 enregistrant
            le modèle.
        intents_path : string
            Le chemin du fichier json stockant les intentions, dans un format
            adapét à patterns.

        See also
        --------
        patterns : module de pré-traitement des intentions.

        Examples
        --------
        >>> model = ChatBotBrain()
        >>> model.train("mymodel", "myintents.json")
        >>> # génère mymodel.h5 et myintents.pkl
        """
        x,y = self._patterns.convert_intents(intents_path)
        self._learn(x, y, model_name)

    def _learn(self, x, y, model_name):
        """
        Créé et compile le modèle, fit les données, et sauvegarde le modèle.

        Parameters
        ----------
        x : list
            La liste des correspondances des patterns (booléens)
        y : list
            La liste des correspondances des intentions (booléens)
        model_name : string
            Le nom du modèle, servant de base du nom de fichier h5 enregistrant
            le modèle.

        See also
        --------
        patterns : module de pré-traitement des intentions.
        """
        # on créé le modèle, à 3 couches :
        #   - 256 neurones
        #   - 128 neurones
        #   - nombre d'intentions
        self._model.add(Dense(256, input_shape=(len(x[0]),), activation="relu"))
        self._model.add(Dropout(0.25))
        self._model.add(Dense(128, activation="relu"))
        self._model.add(Dropout(0.15))
        self._model.add(Dense(len(y[0]), activation="softmax"))

        # compilation du modèle et fitting. Le nombre d'epochs doit être élevé
        # pour plus de précision.
        # Référence :
        # https://towardsdatascience.com/
        #   deep-learning-for-nlp-creating-a-chatbot-with-keras-da5ca051e051
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self._model.compile(
                loss="categorical_crossentropy",
                optimizer=sgd,
                metrics=["accuracy"]
                )
        fit = self._model.fit(
                numpy.array(x),
                numpy.array(y),
                epochs=200,
                verbose=self._verbose
                )

        # on sauvegarde le modèle.
        self._model.save(model_name+".h5", fit)

    def answer(self, sentence):
        """
        Renvoie une réponse à la phrase donnée prédite selon le modèle.

        Parameters
        ----------
        sentence : string
            Une phrase à laquelle le modèle doit donner une réponse.

        Returns
        -------
        string
            Une phrase de réponse.

        Examples
        --------
        >>> model = ChatBotBrain()
        >>> model.train("mymodel", "myintents.json")
        >>> model.answer("Hello World !")
        'Hi !'
        """
        # on prédit l'intention, et on choisit une réponse prédéfinie au hasard.
        return self._patterns.choose_answer(self._which_intent(sentence))

    def _which_intent(self, sentence):
        """
        Prédit l'intention correspondant à la phrase donnée.

        Parameters
        ----------
        sentence : string
            Une phrase dont il faut déterminer l'intention.

        Returns
        -------
        string
            le tag décrivant l'intention prédite.

        See also
        --------
        patterns : module de pré-traitement des intentions.
        """
        pattern = self._patterns.convert_sentence(sentence)
        return self._predict(pattern)

    def _predict(self, pattern):
        """
        Calcule une probabilité pour les différentes intentions du modèle à
        partir du pattern donné, et renvoie la plus haut intention à p > 0.9.

        Parameters
        ----------
        pattern : numpy.array
            Table numpy de booléen indiquant quel lemmas du modèle sont à
            considérer.

        Returns
        -------
        string
            le tag décrivant l'intention prédite.

        See also
        --------
        patterns : module de pré-traitement des intentions.
        """
        # predict renvoie une liste de liste de probabilités
        results = list(self._model.predict(numpy.array([pattern]))[0])
        # On récupère la plus haut probabilité, mais on ne la considère que si
        # elle dépasse un certain seuil (ici 0.9).
        index  = results.index(max(results))
        # on renvoie le tag d'intention prédit.
        return index if results[index] > 0.9 else -1

if __name__ == "__main__":
    pass
