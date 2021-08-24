#!/usr/bin/env python3

# wschatbot version 0.1
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
# Nom     : wschatbot
# Rôle    : programme de chatbot. Ce programme gère l'interaction avec un
#           chatbot basé sur une IA via websocket.
# Auteur  : Martos Alexandre
# E-mail  : alexandre.martos@protonmail.ch
# Version : 0.1
# Licence : GPLv3
# Python  : Python 3.9.2
# Usage   :
#     afficher l'aide               : python wschatbot.py -h / brain --help
#     afficher la licence           : python wschatbot.py -w / brain --license
#     afficher le numéro de version : python wschatbot.py -v / --version
#     ouvrir un websocket           : python wschatbot.y -m model.h5 intents.pkl -p 1234
# *****************************************************************************

import os, sys
import datetime

# librairie gérant l'asynchronisme.
import asyncio
# la librairie gérant le websocket.
import websockets
# une librairie permettant de sécuriser les messages entrants.
import html

from chatbot import ChatBotBrain

async def answer(websocket, path):
    """
    Fonction appelée par le websocket lors de la connexion du client.

    Parameters
    ----------
    Les paramètres sont ceux nécessaires pour faire fonctionner le websocket.
    Pour plus d'informations, utiliser la commande :
    `python -c "import websockets; help(websockets.serve)"`
    """
    while True:
        try:
            # on reçoit un message, et on détermine la réponse.
            msg = await websocket.recv()
            ans = await treat(msg)
            await websocket.send(ans) # on envoie la réponse
        except websockets.exceptions.ConnectionClosed:
            # on élimine les message d'erreurs de déconnexion du client.
            pass

async def treat(msg):
    """
    Traite le message reçu par le client.

    Parameters
    ----------
    msg : string
        Le message reçu.

    Returns
    -------
    str
        La réponse prédite par le chatbot.

    Examples
    --------
    >>> await treat("Hello World !")
    'Hi !'
    """
    # on sécurise tout d'abord le message.
    msg = sanitize(msg)
    return model.answer(msg)

def sanitize(msg):
    """
    Sécurise un message pour une page html, et limite le message à 255
    caractères.

    Parameters
    ----------
    msg : string
        Le message à sécuriser.

    Returns
    -------
    str
        Le message tronqué à 255 caractères et dont les caractères ont été
        remplacés par des entités html (et les anti-slashs supprimés).

    Examples
    --------
    >>> sanitize(" <script>alert(boom!)</script>\n")
    '&lt;script&gt;alert(boom!)&lt;/script&gt;'
    """
    if msg:
        msg = msg.strip()[:256]
        msg = msg.replace("\\", "")
        return html.escape(msg)
    return ""

if __name__ == "__main__":
    import argparse

    # on définit la documentation du programme, et les arguments qu'il prend (à
    # l'aide de la librairie argparse.
    progname = "wschatbot"
    author = "Alexandre Martos"
    license = """{progname} Copyright (C) {year} {author}

This program comes with ABSOLUTELY NO WARRANTY; for details type `{progname} -w'.
This is free software, and you are welcome to redistribute it under certain
conditions; see <https://www.gnu.org/licenses/> for details.""".format(
        progname=progname, year=datetime.datetime.now().year, author=author)
    version = "0.1"

    parser = argparse.ArgumentParser(
            description="""
Ouvre un websocket permettant d'envoyer des phrases au et recevoir des
prédictions du programme brain.
""",
            add_help=False,
            formatter_class=argparse.RawTextHelpFormatter
            )

    parser.add_argument(
            "-h", "--help",
            action="help",
            help="affiche cette aide"
    )
    parser.add_argument(
            "-v", "--version",
            action="store_true",
            help="affiche le numéro de version"
    )
    parser.add_argument(
            "-w", "--license",
            action="store_true",
            help="affiche le texte de licence"
    )
    parser.add_argument(
            "-m", "--model",
            nargs=2,
            type=str,
            help="""(obligatoire) charge un modèle à partir du fichier h5 du
modèle et du fichier pickle des intentions."""
    )
    parser.add_argument(
            "-p", "--port",
            type=int,
            help="(obligatoire) le port du websocket"
    )
    args = parser.parse_args()

    # si l'utilisateur veut afficher la documentation, on l'affiche et on arrête
    # le programme.
    if args.license:
        print(license)
        exit(0)
    elif args.version:
        print(version)
        exit(0)

    # on vérifie que les arguments obligatoires ont été donnés.
    if not (args.model and args.port):
        parser.print_help()
        print()
        raise ValueError(
            "Les arguments --model et --port sont obligatoires.")
    else:
        # on limite les ports à ceux > 1023, ports inférieurs sont réservés
        # voir : https://fr.wikipedia.org/wiki/Liste_de_ports_logiciels
        if not args.port > 1023:
            raise ValueError("Les ports < 1024 sont réservés")

        # on traite les arguments donnés.
        model_path, pickle_path = args.model
        model = ChatBotBrain(model_path, pickle_path)
        # on lance le websocket dirigé vers le localhost. Il faut paramétrer le
        # serveur pour qu'il redirige les requête WS de la page vers le localhost
        # (plus sécurisé que de faire via ip:port directement dans la page).
        ws = websockets.serve(answer, "0.0.0.0", args.port)
        asyncio.get_event_loop().run_until_complete(ws)
        asyncio.get_event_loop().run_forever()
