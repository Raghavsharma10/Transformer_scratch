def npartial(func, *args, **kwargs):
  """
  Returns a partial node visitor function
  """
  def wrapped(self, node):
    func(self, *args, **kwargs)
  return wrapped