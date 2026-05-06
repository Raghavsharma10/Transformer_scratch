def get_src_folder(self):
    """
    Gets the app source folder from settings.gradle file.

    Returns:
      A string containing the project source folder name (default is "app")
    """
    with open('%s/settings.gradle' % self.path) as f:
      for line in f.readlines():
        if line.startswith('include'):
          matches = re.findall(r'\'\:?(.+?)\'', line)
        if len(matches) == 0:
          continue
        for folder in matches:
          if self.is_app_folder(folder):
            return folder
    return 'app'