def find_deref_completions(ast_rootpath, root_env=gcl.default_env):
  """Returns a dict of { name => Completions }."""
  with util.LogTime('find_deref_completions'):
    tup = inflate_context_tuple(ast_rootpath, root_env)
    path = path_until(ast_rootpath, is_deref_node)
    if not path:
      return {}
    deref = path[-1]
    haystack = deref.haystack(tup.env(tup))
    if not hasattr(haystack, 'keys'):
      return {}
    return {n: get_completion(haystack, n) for n in haystack.keys()}