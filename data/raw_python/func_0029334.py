def find_top_level_directory(start_directory):
    """Finds the top-level directory of a project given a start directory
    inside the project.

    Parameters
    ----------
    start_directory : str
        The directory in which test discovery will start.

    """
    top_level = start_directory
    while os.path.isfile(os.path.join(top_level, '__init__.py')):
        top_level = os.path.dirname(top_level)
        if top_level == os.path.dirname(top_level):
            raise ValueError("Can't find top level directory")
    return os.path.abspath(top_level)