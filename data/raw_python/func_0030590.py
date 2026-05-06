def _compose(self, name, args, mkdir=True):
        """Get a named filesystem entry, and extend it into a path with additional
        path arguments"""
        from os.path import normpath
        from ambry.dbexceptions import ConfigurationError

        root = p = self._config.filesystem[name].format(root=self._root)

        if args:
            args = [e.strip() for e in args]
            p = join(p, *args)

        if not isdir(p) and mkdir:
            makedirs(p)

        p = normpath(p)

        if not p.startswith(root):
            raise ConfigurationError("Path for name='{}', args={} resolved outside of define filesystem root"
                                 .format(name, args))

        return p