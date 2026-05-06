def get_package_version(verfile):
  '''Scan the script for the version string'''
  version = None
  with open(verfile) as fh:
      try:
          version = [line.split('=')[1].strip().strip("'") for line in fh if \
              line.startswith('__version__')][0]
      except IndexError:
          pass
  return version