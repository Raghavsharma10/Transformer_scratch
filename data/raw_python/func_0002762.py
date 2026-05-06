def evaluate_relative_path(working=os.getcwd(), relative=''):
    """
    This function receives two strings representing system paths. The first is
    the working directory and it should be an absolute path. The second is the
    relative path and it should not be absolute. This function will render an
    OS-appropriate absolute path, which is the normalized path from working
    to relative.
    """
    return os.path.normpath(os.path.join(working, relative))