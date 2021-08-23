#!/usr/bin/env python3

# patterns version 0.1
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
# Nom     : patterns
# Rôle    : convertit des intentions stockées au format json en un
#           objet utilisable par le module brain. Le format
#           d'intensions est identique à :
#               - https://github.com/jerrytigerxu/Simple-Python-Chatbot
# Auteur  : Martos Alexandre
# E-mail  : alexandre.martos@protonmail.ch
# Version : 0.1
# Licence : GPLv3
# Python  : Python 3.9.2
# Usage   :
#     convertir un fichier d'intentions :
#         Patterns().convert_intents("intentions.json")
#     convertir une phrase en une réponse selon les intentions :
#         patterns = Patterns("intentions.pkl")
#         nparr = patterns.convert_sentence("Hello World !")
#         index  = (prédiction par brain donnant l'index du tag)
#         tag = patterns.get_tag_by_index(index)
#         answer = patterns.choose_answer(tag)
# *****************************************************************************

import os
import random

# une librairie de gestion des fichiers json.
import json
# cette librairie sauvegarde des objets python dans un fichier, qui peuvent être
# rechargés par la suite.
import pickle
# une librairie de gestion de langage, permet de séparer correctement les mots
# des phrases et de raciniser (stemming) les mots.
import nltk
from nltk.stem.snowball import FrenchStemmer
# la seule fonction de numpy utilisée, pour créer des tables.
from numpy import array as nparray

class Patterns:
    """
    Classe permettant de convertir des phrases en un format utilisable par une
    IA gérée par la librairie Keras.

    Methods
    -------
    convert_intents(path)
        Convertit les données d'intentions stockées dans le fichier json indiqué
        en données utilisables pour l'entraînement de l'IA.
    convert_sentence(sentence)
        Renvoie une table numpy utilisable par l'IA pour prédire l'intention de
        la phrase donnée.
    get_tag_by_index(index)
        Renvoie le tag décrivant l'intention correspondant à l'index généré par
        la prédiction de l'IA.
    choose_answer(intent)
        Renvoie une réponse appropriée à l'intention indiquée.

    Notes
    -----
    La structure d'un fichier d'intentions est affichée ci-dessous. Le tag
    "noanswer" est obligatoire (il décrit les réponses de base lorsque la
    prédiction a échoué). Les autres tags sont libres. Le contexte n'est pour le
    moment pas supporté.

    {
        "intents": [
            {
                "tag": "noanswer",
                "patterns": [],
                "responses": ["Je n'ai pas compris..."],
                "context": []
            },
            {
                "tag": "hello_intent",
                "patterns": ["Hello !", "hi"],
                "responses": ["Hey you !", "Hello."],
                "context": ["other_intent"]
            },
            ... (other intents)
        ]
    }

    Examples
    --------
    >>> p = Patterns()
    >>> training_data = p.convert_intents("intents.json")
    >>> # training_data contient les données à fournir à l'IA via Keras.
    >>> # le fichier intents.pkl est créé et peut être utilisé pour recharger
    >>> # l'objet _patterns.
    >>> predict_data = p.convert_sentence("Hello World !")
    >>> # predict_data contient les données à fournir à l'IA via Keras.
    >>> p.get_tag_by_index(0)
    'noanswer'
    >>> p.choose_answer("noanswer")
    'Je n'ai pas compris...'
    """

    def __init__(self, path=""):
        nltk.download("punkt")
        nltk.download("wordnet")
        nltk.download("stopwords")
        self._stemmer = FrenchStemmer()
        self._stoplist = nltk.corpus.stopwords.words("french")

        if path:
            if not os.path.isfile(path):
                raise FileNotFoundError(path)
            with open(path, "rb") as f:
                self._patterns = pickle.load(f)
        else:
            self._patterns = {
                    "tags": {},
                    "stems": [],
            }

    def convert_intents(self, path):
        """
        Convertit les intentions du fichier json donné en un format utilisable
        par une IA gérée par la librairie Keras, et sauvegarde l'objet _patterns
        courant dans un fichier pickle.

        Parameters
        ----------
        path : str
            Le chemin du fichier json d'intentions.

        Returns
        -------
        tuple
            Un tuple de listes contenant des booléens :
                - la liste des correspondances des patterns
                - a liste des correspondances des intentions

        Examples
        --------
        >>> patterns = Patterns()
        >>> patterns.convert_intents("intents.json")
        (
            [
                [True, False, False, ...],
                [...],
                ...
            ], 
            [
                [False, False, False, ...],
                [...],
                ...
            ]
        )
        """
        self._parse_intents_file(path)
        self._sort_all_stems()
        # on récupère le nom de fichier.
        basename = ".".join(path.split(".")[:-1])
        basename = basename.split("/")[-1]
        self._save(self._patterns, basename+".pkl")
        return self._make_training_data()

    def _parse_intents_file(self, path):
        """
        Traite les données du fichier json d'intentions donné.

        Parameters
        ----------
        path : str
            Le chemin du fichier json d'intentions.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        with open(path, "r") as f:
            intents = json.loads(f.read())
        for intent in intents["intents"]:
            if intent:
                self._set_intent(intent)

    def _set_intent(self, intent):
        """
        Stocke les données de l'intention dans l'attribut _patterns.

        Parameters
        ----------
        intent : dict
            Un dictionnaire contenant une description de l'intention.
        """
        tag = intent["tag"]
        if not tag in self._get_all_tags():
            self._patterns["tags"][tag] = {"stems": [],"answers": []}

        self._set_answers(tag, intent["responses"])
        stems = [self._stemming(p) for p in intent["patterns"]]
        self._set_stems(tag, stems)

    def _get_all_tags(self):
        """
        Renvoie l'ensemble des clés le l'attribut _patterns["tag"].

        Returns
        -------
        list
            Une liste ordonnée de tous les tags de l'attribut _patterns.
        """
        return sorted(list(self._patterns.get("tags").keys()))

    def _set_answers(self, tag, answers):
        """
        Stocke les réponses données dans la valeur de la clé correspondant au
        tag dans l'attribut _patterns.

        Parameters
        ----------
        tag : str
            Un tag décrivant une intention.
        answers : list
            Une liste de réponses à l'intention.
        """
        self._patterns["tags"][tag]["answers"] += answers

    def _get_answers(self, tag):
        """
        Renvoie les réponses associées au tag (décrivant une intention) donné.

        Parameters
        ----------
        tag : str
            Un tag décrivant une intention.

        Returns
        -------
        list
            Une liste de réponse à l'intention.
        """
        return self._patterns.get("tags").get(tag).get("answers")

    def _stemming(self, pattern):
        """
        Tranforme le pattern (== phrase) donné en une liste de radicaux ou
        racines. Voir https://fr.wikipedia.org/wiki/Racinisation

        Parameters
        ----------
        pattern : str
            Une phrase correspondant à une intention.

        Returns
        -------
        list
            La liste des racines correspondantes au pattern donné.
        """
        tokenizer = nltk.RegexpTokenizer(r"\w+")
        tokens = tokenizer.tokenize(pattern)
        return [self._stemmer.stem(word.lower()) for word in tokens if not word in self._stoplist]

    def _set_stems(self, tag, stems):
        """
        Enregistre les racines dans l'attribut _patterns.

        Parameters
        ----------
        tag : str
            Le tag décrivant l'intention et correspondant aux racines.
        stems : list
            Une liste des stems correspondant à l'intention.
        """
        tag_stems = self._get_stems(tag)
        all_stems = self._get_all_stems()
        for l in stems:
            for w in l:
                if not w in tag_stems:
                    self._patterns["tags"][tag]["stems"].append(w)
                if not w in all_stems:
                    self._patterns["stems"].append(w)

    def _get_stems(self, tag):
        """
        Renvoie une liste des raince associé au tag (à l'intention) donné.

        Parameters
        ----------
        tag : str
            Un tag décrivant une intention.

        Returns
        -------
        list
            Une liste de toutes les rainces associés à l'intention.
        """
        return self._patterns.get("tags").get(tag).get("stems")

    def _get_all_stems(self):
        """
        Renvoie toutes les racines enregistrés dans l'instance.

        Returns
        -------
        list
            Une liste de toutes les racines de l'instance.
        """
        return self._patterns.get("stems")

    def _sort_all_stems(self):
        self._patterns["stems"] = sorted(set(self._get_all_stems()))
        for tag in self._patterns["tags"].keys():
            self._patterns["tags"][tag]["stems"] = sorted(set(self._get_stems(tag)))

    def _save(self, var, path):
        """
        Sauvegarde l'objet var dans un fichier pickle.

        Parameters
        ----------
        var : string
            Le nom de l'objet à enregistrer.
        path : string
            le chemin du fichier où sauvegarde l'objet.
        """
        with open(path, "wb") as f:
            pickle.dump(var, f)

    def _make_training_data(self):
        """
        Convertit la correspondance stems <> intentions en un format utilisable
        par une IA gérée par la librairie Keras.
        """
        training_data = []
        all_tags = self._get_all_tags()
        all_stems = self._get_all_stems()
        tags_false = [False] * len(all_tags)
        stems_false = [False] * len(all_stems)
        for tag in all_tags:
            for stem in self._get_stems(tag):
                stems_bin = list(stems_false)
                stems_bin[all_stems.index(stem)] = True
                tags_bin = list(tags_false)
                tags_bin[all_tags.index(tag)] = True
                training_data.append([stems_bin, tags_bin])
        random.shuffle(training_data)
        training_data = nparray(training_data, dtype=object)
        return list(training_data[:,0]), list(training_data[:,1])

    def convert_sentence(self, sentence):
        """
        Convertit une phrase en une table numpy indiquant quelles racines
        stockées sont dans la phrase.

        Parameters
        ----------
        sentence : str
            Une phrase à analyser.

        Returns
        -------
        numpy.array
            Une table contenant des booléens.

        Examples
        --------
        >>> patterns = Patterns()
        >>> patterns.convert_intents("intents.json")
        >>> patterns.convert_sentence("Hello World !")
        array([True, False, False, ...])
        """
        stems = self._stemming(sentence)
        stems_bin = [True if w in stems else False for w in self._get_all_stems()]
        return nparray(stems_bin)

    def choose_answer(self, index):
        """
        Renvoie une réponse correspondant à l'intention indiquée.

        Parameters
        ----------
        index : int
            L'index de l'intention à laquelle il faut répondre.

        Returns
        -------
        str
            Une phrase correspondant à l'intention.

        Examples
        --------
        >>> patterns = Patterns()
        >>> patterns.convert_intents("intents.json")
        >>> patterns.choose_answer(-1) # == noanswer
        'Pardon, je n'ai pas compris...'
        >>> patterns.choose_answer(0) # noanswer
        'Pouvez-vous répéter ?'
        >>> patterns.choose_answer(1) # hello_intent
        'Bonjour !'
        """
        intent = self._get_tag_by_index(index)
        return random.choice(self._get_answers(intent))

    def _get_tag_by_index(self, index):
        """
        Renvoie le tag correspondant à l'index donné.

        Parameters
        ----------
        index : int
            L'index du tag à récupérer.

        Returns
        -------
        str
            Le tag décrivant l'intention.
        """
        return self._get_all_tags()[index] if index >= 0 else "noanswer"

if __name__ == "__main__":
    pass
