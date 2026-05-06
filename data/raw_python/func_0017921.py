def enumerate_scope(ast_rootpath, root_env=None, include_default_builtins=False):
  """Return a dict of { name => Completions } for the given tuple node.

  Enumerates all keys that are in scope in a given tuple. The node
  part of the tuple may be None, in case the binding is a built-in.
  """
  with util.LogTime('enumerate_scope'):
    scope = {}
    for node in reversed(ast_rootpath):
      if is_tuple_node(node):
        for member in node.members:
          if member.name not in scope:
            scope[member.name] = Completion(member.name, False, member.comment.as_string(), member.location)

    if include_default_builtins:  # Backwards compat flag
      root_env = gcl.default_env

    if root_env:
      for k in root_env.keys():
        if k not in scope and not hide_from_autocomplete(root_env[k]):
          v = root_env[k]
          scope[k] = Completion(k, True, dedent(v.__doc__ or ''), None)

    return scope