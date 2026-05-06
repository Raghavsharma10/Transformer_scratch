def prepare_env(cls):
        """Prepares current environment and returns Python binary name.

        This adds some virtualenv friendliness so that we try use uwsgi from it.

        :rtype: str|unicode
        """
        os.environ['PATH'] = cls.get_env_path()
        return os.path.basename(Finder.python())