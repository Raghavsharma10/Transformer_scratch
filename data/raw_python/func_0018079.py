def loads(s, filename=None, loader=None, implicit_tuple=True, env={}, schema=None):
  """Load and evaluate a GCL expression from a string."""
  ast = reads(s, filename=filename, loader=loader, implicit_tuple=implicit_tuple)
  if not isinstance(env, framework.Environment):
    # For backwards compatibility we accept an Environment object. Otherwise assume it's a dict
    # whose bindings will add/overwrite the default bindings.
    env = framework.Environment(dict(_default_bindings, **env))
  obj = framework.eval(ast, env)
  return mod_schema.validate(obj, schema)