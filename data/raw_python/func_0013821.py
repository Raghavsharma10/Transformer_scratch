def get_default_cache():
    """Get the default cache location.

    Adheres to the XDG Base Directory specification, as described in
    https://standards.freedesktop.org/basedir-spec/basedir-spec-latest.html

    :return: the default cache directory absolute path
    """

    xdg_cache_home = os.environ.get('XDG_CACHE_HOME') or \
        os.path.join(os.path.expanduser('~'), '.cache')

    return os.path.join(xdg_cache_home, 'pokebase')