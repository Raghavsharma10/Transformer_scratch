def handler(self, path='/app', ctx='all'):
    """
    Handler that prints the project build log into the STDOUT (using the ``project`` package).

    :param path(str): the project source code path, default is '/app'.
    :param ctx(str): build log context file to be used, available options: validate, prepare, build or all.

    Returns:
      Doesn't return a value but prints the output in the STDOUT/STDERR (separated by a comma).
    """
    options = {
      'path': path
    }
    project = builds.from_path(path)
    project.log(ctx=ctx)