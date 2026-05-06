def from_path(cls, path):
    """
    Instantiates a project class from a given path.

    :param path: app folder path source code

    Returns
      A project instance.
    """
    if os.path.exists(path) is False:
      raise errors.InvalidPathError(path)
    return cls(path=path)