def handler(self, path='/app'):
    """
    Handler that prints the apk file(s) path that can be exported from the container (using the ``project`` package).

    :param path(str): the project source code path, default is '/app'.

    Returns:
      Doesn't return a value but prints the output in the STDOUT/STDERR (separated by a comma).
    """
    options = {
      'path': path
    }
    project = builds.from_path(path)
    print(project.get_export_path())