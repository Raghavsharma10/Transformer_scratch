def is_app_folder(self, folder):
    """
    checks if a folder 
    """
    with open('%s/%s/build.gradle' % (self.path, folder)) as f:
      for line in f.readlines():
        if config.gradle_plugin in line:
          return True
    return False