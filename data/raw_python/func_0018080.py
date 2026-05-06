def load(filename, loader=None, implicit_tuple=True, env={}, schema=None):
  """Load and evaluate a GCL expression from a file."""
  with open(filename, 'r') as f:
    return loads(f.read(),
                 filename=filename,
                 loader=loader,
                 implicit_tuple=implicit_tuple,
                 env=env,
                 schema=schema)