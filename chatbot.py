#!/usr/bin/env python3

# chatbot version 0.1
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
# Nom               : chatbot
# Rôle              : programme de chatbot. Ce programme gère la création d'un
#                     modèle de prédiction de réponses et la prédiction
#                     elle-même à l'aide des modules brain et patterns. Un
#                     fichier décrivant les différentes intentions est utilisé
#                     comme données de base. Un système de chat de base est
#                     disponible via la ligne de commandes.
#                     Inspiré de :
#                       - https://github.com/jerrytigerxu/Simple-Python-Chatbot
#                       - https://databotlab.com/creating-a-neural-network-chatbot-using-tensorflow-and-nltk/
# Auteur            : Martos Alexandre
# E-mail            : alexandre.martos@protonmail.ch
# Version           : 0.1
# Licence           : GPLv3
# Python            : Python 3.9.2
# Usage (programme) :
#     afficher l'aide               : python chatbot.py -h / --help
#     afficher la licence           : python chatbot.py -w / --license
#     afficher le numéro de version : python chatbot.py -v / --version
#     générer un modèle             : python chatbot.py -n nom intentions.json
#     ouvrir un chat avec le bot    : python chatbot.py -l nom.h5 intentions.pkl
# *****************************************************************************

if __name__ == "__main__":
    import argparse, datetime
    from chatbot import ChatBotBrain

    # on définit la documentation du programme, et les arguments qu'il prend (à
    # l'aide de la librairie argparse.
    progname = "chatbot"
    author = "Alexandre Martos"
    license = """{progname} Copyright (C) {year} {author}

This program comes with ABSOLUTELY NO WARRANTY; for details type `{progname} -w'.
This is free software, and you are welcome to redistribute it under certain
conditions; see <https://www.gnu.org/licenses/> for details.""".format(
        progname=progname, year=datetime.datetime.now().year, author=author)
    version = "0.1"

    parser = argparse.ArgumentParser(
            description="""{progname} version {version}

Chatbot géré par une IA. Le modèle, créé à partir d'un fichier json
d'intentions, est préalablement traité via l'option -n/--new_model pour générer
le modèle. L'option -l/--load_model permet de lancer un chat et de charger un
modèle (et les intentions) pour prédire les réponses.""".format(
                progname=progname, version =version),
            add_help=False,
            # on a besoin de nouvelles lignes pour le format du json, donc on
            # change de formatteur de texte (argparse supprimes les lignes par
            # défaut).
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
            "-n", "--new_model",
            nargs=2,
            type=str,
            help="""créé un nouveau modèle à partir du nom et chemin de fichier
json donné (contenant les intentions au format requis par patterns)"""
    )
    parser.add_argument(
            "-l", "--load_model",
            nargs=2,
            type=str,
            help="""charge un modèle à partir du fichier h5 du modèle et du
fichier pickle des intentions."""
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

    # si l'utilisateur veut utiliser le programme directement (sans utiliser
    # uniquement la classe), il a deux manières de le faire : en utilisant un
    # modèle préexistant (auquel cas il active un chat d'échange avec le "bot"),
    # ou en créant un nouveau modèle (le chat ne s'active pas).
    if args.load_model:
        # on charge le modèle indiqué
        model_path, pickle_path = args.load_model
        model = ChatBotBrain(model_path, pickle_path)
        # une interface de chat extrêmement simple.
        while(True):
            print("you > ", end="")
            print("bot >", model.answer(input()))
    elif args.new_model:
        # on créé un modèle à partir des intentions données.
        model_name, intents_path = args.new_model
        # on vérifie tout d'abord que le nom est valide. Le fichier json est
        # vérifié par le module patterns.
        if not model_name:
            raise ValueError("le modèle doit avoir un nom")
        model = ChatBotBrain()
        ChatBotBrain().train(model_name, intents_path)
    else:
        parser.print_help()
        print()
        raise ValueError("pas d'arguments fournis")
