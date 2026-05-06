def baseglob(pat, base):
    """Given a pattern and a base, return files that match the glob pattern
    and also contain the base."""
    return [f for f in glob(pat) if f.startswith(base)]