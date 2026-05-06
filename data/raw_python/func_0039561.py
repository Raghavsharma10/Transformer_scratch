def prepare(self):
    """
    Prepares the android project to the build process.

    Prepare the ant project to be built
    """
    cmd = [
      'android', 'update', 'project',
      '-p', self.path,
      '-t', self.get_target()
    ]
    self.run_cmd(cmd, 'prepare')