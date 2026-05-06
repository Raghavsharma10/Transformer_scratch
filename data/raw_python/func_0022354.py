def answer(part, module='mlai2014.json'):
    """Returns the answers to the lab classes."""
    marks = json.load(open(os.path.join(data_directory, module), 'rb'))
    return marks['Lab '  + str(part+1)]