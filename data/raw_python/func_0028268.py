def print_to_file(object_name):
    """
    Function takes in object of type str, list, or dict and prints out to current working
    directory as pyoutput.txt
    :param:  Object: object of type str, list, or dict
    :return: No return. Just prints out to file handler and save to current working directory as
    pyoutput.txt
    """
    with open('pyoutput.txt', 'w') as filehandler:
        output = None
        if isinstance(object_name, list):
            output = json.dumps(object, indent=4)
        if isinstance(object_name, dict):
            output = json.dumps(object, indent=4)
        if isinstance(object_name, str):
            output = object_name
        filehandler.write(output)