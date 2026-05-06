def load(file):
    """ Load events from a json file """
    with open(file, 'r') as f:
        contents = f.read()
    lambder.load_events(contents)