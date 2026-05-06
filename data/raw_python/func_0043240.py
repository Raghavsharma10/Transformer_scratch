def create_env(env_file):
    """Create environ dictionary from current os.environ and
    variables got from given `env_file`"""

    environ = {}
    with open(env_file, 'r') as f:
        for line in f.readlines():
            line = line.rstrip(os.linesep)
            if '=' not in line:
                continue
            if line.startswith('#'):
                continue
            key, value = line.split('=', 1)
            environ[key] = parse_value(value)
    return environ