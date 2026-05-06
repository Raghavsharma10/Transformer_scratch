def check_parameter_similarity(files_dict):
    """
    Checks if the parameter names of all files are similar. Takes the dictionary from get_parameter_from_files output as input.

    """
    try:
        parameter_names = files_dict.itervalues().next().keys()  # get the parameter names of the first file, to check if these are the same in the other files
    except AttributeError:  # if there is no parameter at all
        if any(i is not None for i in files_dict.itervalues()):  # check if there is also no parameter for the other files
            return False
        else:
            return True
    if any(parameter_names != i.keys() for i in files_dict.itervalues()):
        return False
    return True