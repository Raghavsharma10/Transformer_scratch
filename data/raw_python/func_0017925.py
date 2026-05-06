def find_inherited_key_completions(rootpath, root_env):
  """Return completion keys from INHERITED tuples.

  Easiest way to get those is to evaluate the tuple, check if it is a CompositeTuple,
  then enumerate the keys that are NOT in the rightmost tuple.
  """
  tup = inflate_context_tuple(rootpath, root_env)
  if isinstance(tup, runtime.CompositeTuple):
    keys = set(k for t in tup.tuples[:-1] for k in t.keys())
    return {n: get_completion(tup, n) for n in keys}
  return {}