def find_value_at_cursor(ast_tree, filename, line, col, root_env=gcl.default_env):
  """Find the value of the object under the cursor."""
  q = gcl.SourceQuery(filename, line, col)
  rootpath = ast_tree.find_tokens(q)
  rootpath = path_until(rootpath, is_thunk)

  if len(rootpath) <= 1:
    # Just the file tuple itself, or some non-thunk element at the top level
    return None

  tup = inflate_context_tuple(rootpath, root_env)
  try:
    if isinstance(rootpath[-1], ast.Inherit):
      # Special case handling of 'Inherit' nodes, show the value that's being
      # inherited.
      return tup[rootpath[-1].name]
    return rootpath[-1].eval(tup.env(tup))
  except gcl.EvaluationError as e:
    return e