def get_environment_paths(config, env):
    """
    Get environment paths from given environment variable.
    """
    if env is None:
        return config.get(Config.DEFAULTS, 'environment')

    # Config option takes precedence over environment key.
    if config.has_option(Config.ENVIRONMENTS, env):
        env = config.get(Config.ENVIRONMENTS, env).replace(' ', '').split(';')
    else:
        env = os.getenv(env)
        if env:
            env = env.split(os.pathsep)
    return [i for i in env if i]