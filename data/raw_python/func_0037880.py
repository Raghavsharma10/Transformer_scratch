def read_json(file_path):
    """ Read in a json file and return a dictionary representation """
    try:
        with open(file_path, 'r') as f:
            config = json_tricks.load(f)
    except ValueError:
        print('    '+'!'*58)
        print('    Woops! Looks the JSON syntax is not valid in:')
        print('        {}'.format(file_path))
        print('    Note: commonly this is a result of having a trailing comma \n    in the file')
        print('    '+'!'*58)
        raise

    return config