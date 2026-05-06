def handler(self, path='/app', mode='debug'):
    """
    Handler that executes the application build based on its platform (using the ``project`` package).

    param: path(str): the project source code path, default is '/app'.
    param: target(str): the platform target, android-23 is the default for android platform if no value is provided.

    Returns:
      Doesn't return a value but prints the output in the STDOUT/STDERR.
    """
    options = {
      'path': path
    }
    project = builds.from_path(path)
    project.prepare()
    project.validate()
    project.build(mode=mode)