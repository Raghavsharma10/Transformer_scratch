def resolve_file(fname, paths):
  """Resolve filename relatively against one of the given paths, if possible."""
  fpath = path.abspath(fname)
  for p in paths:
    spath = path.abspath(p)
    if fpath.startswith(spath):
      return fpath[len(spath) + 1:]
  return fname