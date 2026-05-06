def matches_factor_conditions(s, env):
    """"Returns True if py{33, 34} expanded is contained in env.name."""
    env_labels = set(env.name.split('-'))
    labels = set(bash_expand(s))
    return bool(labels & env_labels)