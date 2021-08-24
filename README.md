# chatbot

Un chatbot en Python basé sur Keras.

## Installation

Le programme requiert Python 3.9 ou supérieur, le programme `pip` les librairies
listées dans le fichier `requirements.txt`, ainsi que la librairie `keras` et le
backend `theano`.

Les commandes d'installation sur Debian ou dérivé (la commande
`make requirements` effectue toutes les étapes d'installation de prérequis):

```bash
git clone https://github.com/amartos/chatbot.git
cd chatbot
pip install --user -r requirements
sudo apt install -y python3-keras
```

## Utilisation

### Compilation du modèle

La librairie `chatbot.brain` compile le modèle de l'IA à partir d'un fichier
d'intentions json. Ce fichier a la structure suivante:

```json
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
```

**Pour le moment les contextes d'intentions ne sont pas supportés.**

La compilation s'effectue de la manière suivante:

```bash
python chatbot.py --new_model name path/to/my/intents.json
```

La compilation génère, dans le dossier courant, un fichier `name.h5` stockant le
modèle, et un fichier `intents.pkl` stockant les intentions. Ce sont ces deux
fichiers qui sont utilisés par le chatbot par la suite.

### Le chatbot

Un chatbot simple est disponible en ligne de commande. Il est accessible en
lançant:

```bash
python chatbot.py --load_model name.h5 intents.pkl
```

Le chatbot est également utilisable via un websocket:

```bash
python wschatbot.py --model name.h5 intents.pkl --port 1234
```

## Licence

Le programme est sous licence [GPL-v3](LICENSE.md).
