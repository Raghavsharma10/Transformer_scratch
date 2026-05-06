def reads(s, filename=None, loader=None, implicit_tuple=True, allow_errors=False):
  """Load but don't evaluate a GCL expression from a string."""
  return ast.reads(s,
      filename=filename or '<input>',
      loader=loader or default_loader,
      implicit_tuple=implicit_tuple,
      allow_errors=allow_errors)