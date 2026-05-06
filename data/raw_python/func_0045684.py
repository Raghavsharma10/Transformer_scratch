def staticdir():
    """Return the location of the static data directory."""
    root = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(root, "static")