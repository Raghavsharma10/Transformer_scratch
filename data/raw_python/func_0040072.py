def build(self, mode='debug'):
    """
    Builds the app project after the execution of validate and prepare.

    This is the third and last step in the build process.

    Needs to be implemented by the subclass.
    """
    self.ensure_cache_folder()
    ref = {
      'debug': 'assembleDebug',
      'release': 'assembleRelease'
    }
    cmd = [
      './gradlew',
      ref.get(mode, mode),
      '--gradle-user-home',
      self.cache_folder 
    ]
    self.run_cmd(cmd, 'build')