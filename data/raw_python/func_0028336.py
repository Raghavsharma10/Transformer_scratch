def print_to_file(object):
    """
    Function takes in object of type str, list, or dict and prints out to current working directory as pyoutput.txt
    :param:  Object: object of type str, list, or dict
    :return: No return. Just prints out to file handler and save to current working directory as pyoutput.txt
    """
    with open ('pyoutput.txt', 'w') as fh:
        x = None
        if type(object) is list:
            x = json.dumps(object, indent = 4)
        if type(object) is dict:
            x = json.dumps(object, indent = 4)
        if type (object) is str:
            x = object
        fh.write(x)