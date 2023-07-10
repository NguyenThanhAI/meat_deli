import json
import os

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
config_file = os.path.join(THIS_DIR, 'data_main', 'config.json')
USER_DIR = os.path.expanduser('~')
DATABASE_DIR = os.path.join(USER_DIR, 'Aimesoft')


with open(config_file) as fi:
    data = json.load(fi)


def reload():
    global data
    with open(config_file) as fi:
        data = json.load(fi)
