def version(cls):  # noqa: N805  # pylint: disable=no-self-argument
        """
        :py:class:Returns `str` -- Returns :attr:`_version_` if set,
        otherwise falls back to module `__version__` or None
        """
        return cls._version_ or getattr(sys.modules.get(cls.__module__, None),
                                        '__version__', None)