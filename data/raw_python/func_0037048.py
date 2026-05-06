def read_config(config):
    """Read config file and return uncomment line
    """
    for line in config.splitlines():
        line = line.lstrip()
        if line and not line.startswith("#"):
            return line
    return ""