#!/usr/bin/env python3

# test_patterns version 0.1
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
# Nom     : test_patterns
# Rôle    : unité de test du module chatbot.patterns
# Auteur  : Martos Alexandre
# E-mail  : alexandre.martos@protonmail.ch
# Version : 0.1
# Licence : GPLv3
# Python  : Python 3.9.2
# Usage   : python -m unittest tests/units/test_patterns.py
# *****************************************************************************

# une librairie standard de tests unitaires
import unittest

import os
import json
from numpy import array

# le module testé
from chatbot import Patterns

class patternsTest(unittest.TestCase):
    """
    Classe de test unitaire du module chatbot.patterns.

    Methods
    -------
    test_A_convert_intent()
        Teste le traitement et la conversion correcte d'un fichier d'intentions
        en un dictionnaire d'intentions et en données utilisables par Keras.

    test_B_convert_sentences()
        Teste la conversion de phrase en données utilisables par Keras.

    test_C_get_tag_by_index()
        Teste la récupération du tag d'intention correct via l'index donné.

    test_D_choose_answer()
        Teste la capacité de choisir correctement une réponse associée au tag
        donné.
    """

    # on redéfinit l'init car on a besoin de l'instance de Patterns pour tous
    # les tests.
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._p = Patterns()
        self._p.convert_intents("tests/assets/test_intents.json")

    # l'ordre des fonction appelées est alphabétique.
    def test_A_convert_intent(self):
        """
        Teste le traitement et la conversion correcte d'un fichier d'intentions
        en un dictionnaire d'intentions et en données utilisables par Keras.
        """
        # on charge le dictionnaire que l'on devrait obtenir.
        with open("tests/assets/test_intents_converted.json", "r") as f:
            a = json.load(f)
        self.assertTrue(os.path.isfile("test_intents.pkl"))
        self.assertEqual(self._p._patterns, a)

    def test_B_convert_sentences(self):
        """
        Teste la conversion de phrase en données utilisables par Keras.
        """
        a = self._p.convert_sentence("bien le bonjour !")
        b = array([True, True, False, True, False, False])
        # unittest a du mal à comparer deux tables numpy sans .any ou .all
        self.assertEqual(a.all(), b.all())

    def test_C_choose_answer(self):
        """
        Teste la capacité de choisir correctement une réponse associée au à
        l'index du tag donné.
        """
        self.assertEqual(
                self._p.choose_answer(-1), # noanswer
                "Désolé, mais je ne comprend pas votre phrase."
            )
        self.assertEqual(
                self._p.choose_answer(0), # noanswer
                "Désolé, mais je ne comprend pas votre phrase."
            )
        self.assertEqual(
                self._p.choose_answer(1), # salutations
                "Bienvenue ! 😊"
            )
