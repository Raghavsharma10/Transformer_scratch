def lock(self):
    """Generates (and returns) a blocking session object to the database."""
    if hasattr(self, 'session'):
      raise RuntimeError('Dead lock detected. Please do not try to lock the session when it is already locked!')

    if LooseVersion(sqlalchemy.__version__) < LooseVersion('0.7.8'):
      # for old sqlalchemy versions, in some cases it is required to re-generate the engine for each session
      self._engine = sqlalchemy.create_engine("sqlite:///"+self._database)
      self._session_maker = sqlalchemy.orm.sessionmaker(bind=self._engine)

    # create the database if it does not exist yet
    if not os.path.exists(self._database):
      self._create()

    # now, create a session
    self.session = self._session_maker()
    logger.debug("Created new database session to '%s'" % self._database)
    return self.session