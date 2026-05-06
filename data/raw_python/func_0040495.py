def log(self, ctx='all'):
    """
    Gets the build log output.

    :param ctx: specifies which log message to show, it can be 'validate', 'build' or 'all'.
    """
    path = '%s/%s.log' % (self.path, ctx)
    if os.path.exists(path) is True:
      with open(path, 'r') as f:
        print(f.read())
      return
    validate_path = '%s/validate.log' % self.path
    build_path = '%s/build.log' % self.path
    out = []
    with open(validate_path) as validate_log, open(build_path) as build_log:
      for line in validate_log.readlines():
        out.append(line)
      for line in build_log.readlines():
        out.append(line)
    print(''.join(out))