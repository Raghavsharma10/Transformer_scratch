def get_env(env_file='.env'):
    """
    Set default environment variables from .env file
    """
    try:
        with open(env_file) as f:
            for line in f.readlines():
                try:
                    key, val = line.split('=', maxsplit=1)
                    os.environ.setdefault(key.strip(), val.strip())
                except ValueError:
                    pass
    except FileNotFoundError:
        pass