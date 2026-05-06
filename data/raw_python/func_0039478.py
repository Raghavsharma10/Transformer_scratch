def zipalign(source, dist, build_tool=None, version='4', path=None):
  """
  Uses zipalign based on a provided build tool version (defaulit is 23.0.2).

  :param source(str) - source apk file to be zipaligned
  :param dist(str) - zipaligned apk file path to be created
  :param build_tool(str) - build tool version to be used by zipalign (default is 23.0.2)
  :param version(str) - zipalign version, default is 4
  :param path(str) - basedir to run the command
  """
  if build_tool is None:
    build_tool = config.build_tool_version
  android_home = os.environ.get('AG_MOBILE_SDK', os.environ.get('ANDROID_HOME'))
  cmd_path = [
    android_home,
    '/build-tools',
    '/%s' % build_tool,
    '/zipalign'
  ]
  cmd = [
    ''.join(cmd_path),
    '-v',
    version,
    source,
    dist,
  ]
  common.run_cmd(cmd, log='zipalign.log', cwd=path)