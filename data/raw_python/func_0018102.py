def resolve(self, current_file, rel_path):
    """Search the filesystem."""
    p = path.join(path.dirname(current_file), rel_path)
    if p not in self.file_dict:
      raise RuntimeError('No such fake file: %r' % p)
    return p, p