def resolve(self, current_file, rel_path):
    """Search the filesystem."""
    search_path = [path.dirname(current_file)] + self.search_path

    target_path = None
    for search in search_path:
      if self.exists(path.join(search, rel_path)):
        target_path = path.normpath(path.join(search, rel_path))
        break

    if not target_path:
      raise exceptions.EvaluationError('No such file: %r, searched %s' %
                            (rel_path, ':'.join(search_path)))

    return target_path, path.abspath(target_path)