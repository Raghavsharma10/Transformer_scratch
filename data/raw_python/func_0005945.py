def get_env_path(cls):
        """Returns PATH environment variable updated to run uwsgiconf in
        (e.g. for virtualenv).

        :rtype: str|unicode
        """
        return os.path.dirname(Finder.python()) + os.pathsep + os.environ['PATH']