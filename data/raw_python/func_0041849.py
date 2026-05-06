def ormSessionCreator(self) -> DbSessionCreator:
        """ Get Orm Session

        :return: A SQLAlchemy session scoped for the callers thread..
        """
        assert self._dbConnectString

        if self._ScopedSession:
            return self._ScopedSession

        self._dbEngine = create_engine(
            self._dbConnectString,
            **self._dbEngineArgs
        )

        self._ScopedSession = scoped_session(
            sessionmaker(bind=self._dbEngine))

        return self._ScopedSession