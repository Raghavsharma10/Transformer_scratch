def find_completions_at_cursor(ast_tree, filename, line, col, root_env=gcl.default_env):
  """Find completions at the cursor.

  Return a dict of { name => Completion } objects.
  """
  q = gcl.SourceQuery(filename, line, col - 1)
  rootpath = ast_tree.find_tokens(q)

  if is_identifier_position(rootpath):
    return find_inherited_key_completions(rootpath, root_env)

  try:
    ret = find_deref_completions(rootpath, root_env) or enumerate_scope(rootpath, root_env=root_env)
    assert isinstance(ret, dict)
    return ret
  except gcl.EvaluationError:
    # Probably an unbound value or something--just return an empty list
    return {}