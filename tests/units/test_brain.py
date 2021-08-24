#!/usr/bin/env python3

# test_brain version 0.1
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
# Nom     : test_brain
# Rôle    : unité de test du module chatbot.brain
# Auteur  : Martos Alexandre
# E-mail  : alexandre.martos@protonmail.ch
# Version : 0.1
# Licence : GPLv3
# Python  : Python 3.9.2
# Usage   : python -m unittest tests/units/test_brain.py
# *****************************************************************************

# une librairie standard de tests unitaires
import unittest

import os
from numpy import array

# le module testé
from chatbot import ChatBotBrain

class brainTest(unittest.TestCase):
    """
    Classe de test unitaire du module chatbot.brain.

    Methods
    -------
    test_A_train_and_answer()
        Teste la création du modèle de l'IA à partir d'un fichier d'intentions,
        et les réponses prédite le modèle.

    test_B_load_and_answer()
        Teste la capacité de recharger un modèle et ses intentions, et de
        prédire les réponses via le modèle chargé.
    """

    # l'ordre des fonction appelées est alphabétique.
    def test_A_train_and_answer(self):
        """
        Teste la création du modèle de l'IA à partir d'un fichier d'intentions,
        et les réponses prédite le modèle.
        """
        model = ChatBotBrain(verbose=False)
        # le modèle de test est trop petit pour un test correct
        model.train("test_model", "assets/intents.json")
        self.assertEqual(
                model.answer("comment vas-tu"),
                "Je suis parfaitement fonctionnel. 😉"
        )
        self.assertTrue(
            model.answer("foo") in [
                "Pardonnez-moi, mais je n'ai pas compris.",
                "Désolé, mais je ne comprend pas votre phrase.",
                "Mes excuses, mais pouvez-vous reformuler ?"
            ]
        )

    def test_B_load_and_answer(self):
        """
        Teste la capacité de recharger un modèle et ses intentions, et de
        prédire les réponses via le modèle chargé.
        """
        model = ChatBotBrain("test_model.h5", "intents.pkl")
        self.assertEqual(
                model.answer("quelle est la réponse à tout"),
                "42"
        )
        self.assertTrue(
            model.answer("foo") in [
                "Pardonnez-moi, mais je n'ai pas compris.",
                "Désolé, mais je ne comprend pas votre phrase.",
                "Mes excuses, mais pouvez-vous reformuler ?"
            ]
        )
