def get_env():
    """
    Read environment from ENV and mangle it to a (lower case) representation
    Note: gcdt.utils get_env() is used in many cloudformation.py templates
    :return: Environment as lower case string (or None if not matched)
    """
    env = os.getenv('ENV', os.getenv('env', None))
    if env:
        env = env.lower()
    return env