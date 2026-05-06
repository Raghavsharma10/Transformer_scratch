def get_build_tool_version(self):
    """
    Gets the build tool version to be used by zipalign from build.gradle file.

    Returns:
      A string containing the build tool version, default is 23.0.2.
    """
    with open('%s/%s/build.gradle' % (self.path, self.src_folder)) as f:
      for line in f.readlines():
        if 'buildToolsVersion' in line:
          matches = re.findall(r'buildToolsVersion \"(.+?)\"', line)
          if len(matches) == 1:
            return matches[0]
    return config.build_tool_version