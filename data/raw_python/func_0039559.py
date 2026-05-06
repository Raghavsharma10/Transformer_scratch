def get_target(self):
    """
    Reads the android target based on project.properties file.

    Returns
      A string containing the project target (android-23 being the default if none is found)
    """
    with open('%s/project.properties' % self.path) as f:
      for line in f.readlines():
        matches = re.findall(r'^target=(.*)', line)
        if len(matches) == 0:
          continue
        return matches[0].replace('\n', '')
    return 'android-%s' % (config.sdk_version)