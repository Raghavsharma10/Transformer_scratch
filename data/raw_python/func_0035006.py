def prev_deps(env):
    """Naively gets the dependancies from the last time ctox was run."""
    # TODO something more clever.
    if not os.path.isfile(env.envctoxfile):
        return []

    with open(env.envctoxfile) as f:
        return f.read().split()