def filename_to_module(filename):
    """
    convert a filename like html5lib-0.999.egg-info to html5lib
    """
    find = re.compile(r"^[^.|-]*")
    name = re.search(find, filename).group(0)
    return name