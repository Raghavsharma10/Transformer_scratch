def directory(cls, prefix=None):
        """
        Path that should be used for caching. Different for all subclasses.
        """
        prefix = prefix or utility.read_config().directory
        name = cls.__name__.lower()
        directory = os.path.expanduser(os.path.join(prefix, name))
        utility.ensure_directory(directory)
        return directory