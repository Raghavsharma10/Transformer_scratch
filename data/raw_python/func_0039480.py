def get_highest_build_tool(sdk_version=None):
  """
  Gets the highest build tool version based on major version sdk version.

  :param sdk_version(int) - sdk version to be used as the marjor build tool version context.

  Returns:
    A string containg the build tool version (default is 23.0.2 if none is found)
  """
  if sdk_version is None:
    sdk_version = config.sdk_version
  android_home = os.environ.get('AG_MOBILE_SDK', os.environ.get('ANDROID_HOME'))
  build_tool_folder = '%s/build-tools' % android_home
  folder_list = os.listdir(build_tool_folder)
  versions = [folder for folder in folder_list if folder.startswith('%s.' % sdk_version)]
  if len(versions) == 0:
    return config.build_tool_version
  return versions[::-1][0]