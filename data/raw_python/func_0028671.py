def _default_temporary_directory(self, prefix):
    """__init__ helper."""
    try:
      self.temporary_file = tempfile.NamedTemporaryFile(prefix=prefix)

    except OSError as err: # pragma: no cover
      logger.critical('Cannot create a system temporary directory: '+repr(err))
      raise securesystemslib.exceptions.Error(err)