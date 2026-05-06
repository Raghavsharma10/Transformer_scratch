def find(name=None, ext=None, directory=".", match_case=False,
         disable_glob=False, depth=None):
    """ Designed for the interactive interpreter by making default order
    of find_files faster.

    :param name: Part of the file name
    :param ext: Extensions of the file you are looking for
    :param directory: Top location to recursively search for matching files
    :param match_case: If name has to be a direct match or not
    :param disable_glob: Do not look for globable names or use glob magic check
    :param depth: How many directories down to search
    :return: list of all files in the specified directory
    """
    return find_files_list(directory=directory, ext=ext, name=name,
                           match_case=match_case, disable_glob=disable_glob,
                           depth=depth)