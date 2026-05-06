def _create(self):
    """Creates a new and empty database."""
    from .tools import makedirs_safe

    # create directory for sql database
    makedirs_safe(os.path.dirname(self._database))

    # create all the tables
    Base.metadata.create_all(self._engine)
    logger.debug("Created new empty database '%s'" % self._database)