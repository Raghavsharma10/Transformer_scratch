def get_absolute_path(some_path):
    """
    This function will return an appropriate absolute path for the path it is
    given. If the input is absolute, it will return unmodified; if the input is
    relative, it will be rendered as relative to the current working directory.
    """
    if os.path.isabs(some_path):
        return some_path
    else:
        return evaluate_relative_path(os.getcwd(), some_path)