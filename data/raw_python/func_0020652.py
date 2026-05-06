def load_module(filename):
  """
  Loads a module by filename
  """
  basename = os.path.basename(filename)
  path = os.path.dirname(filename)
  sys.path.append(path)
  # TODO(tlan) need to figure out how to handle errors thrown here
  return __import__(os.path.splitext(basename)[0])