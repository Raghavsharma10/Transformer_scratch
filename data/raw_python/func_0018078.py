def read(filename, loader=None, implicit_tuple=True, allow_errors=False):
  """Load but don't evaluate a GCL expression from a file."""
  with open(filename, 'r') as f:
    return reads(f.read(),
                 filename=filename,
                 loader=loader,
                 implicit_tuple=implicit_tuple,
                 allow_errors=allow_errors)