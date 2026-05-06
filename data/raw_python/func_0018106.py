def compose_all(tups):
  """Compose all given tuples together."""
  from . import ast  # I weep for humanity
  return functools.reduce(lambda x, y: x.compose(y), map(ast.make_tuple, tups), ast.make_tuple({}))