def engine(self):
        """return the SqlAlchemy engine for this database."""

        if not self._engine:

            if 'postgres' in self.driver:

                if 'connect_args' not in self.engine_kwargs:
                    self.engine_kwargs['connect_args'] = {
                        'application_name': '{}:{}'.format(self._application_prefix, os.getpid())
                    }

                # For most use, a small pool is good to prevent connection exhaustion, but these settings may
                # be too low for the main public web application.

                self._engine = create_engine(self.dsn, echo=self._echo,
                                             pool_size=5, max_overflow=5, **self.engine_kwargs)

            else:
                self._engine = create_engine(
                    self.dsn, echo=self._echo, **self.engine_kwargs)

            #
            # Disconnect connections that have a different PID from the one they were created in.
            # This protects against re-use in multi-processing.
            #
            @event.listens_for(self._engine, 'connect')
            def connect(dbapi_connection, connection_record):
                connection_record.info['pid'] = os.getpid()

            @event.listens_for(self._engine, 'checkout')
            def checkout(dbapi_connection, connection_record, connection_proxy):

                from sqlalchemy.exc import DisconnectionError
                pid = os.getpid()
                if connection_record.info['pid'] != pid:

                    connection_record.connection = connection_proxy.connection = None
                    raise DisconnectionError(
                        "Connection record belongs to pid %s, attempting to check out in pid %s" %
                        (connection_record.info['pid'], pid))

            if self.driver == 'sqlite':
                @event.listens_for(self._engine, 'connect')
                def pragma_on_connect(dbapi_con, con_record):
                    """ISSUE some Sqlite pragmas when the connection is created."""

                    # dbapi_con.execute('PRAGMA foreign_keys = ON;')
                    # Not clear that there is a performance improvement.

                    # dbapi_con.execute('PRAGMA journal_mode = WAL')
                    dbapi_con.execute('PRAGMA synchronous = OFF')
                    dbapi_con.execute('PRAGMA temp_store = MEMORY')
                    dbapi_con.execute('PRAGMA cache_size = 500000')
                    if self._foreign_keys:
                        dbapi_con.execute('PRAGMA foreign_keys=ON')

            with self._engine.connect() as conn:
                _validate_version(conn, self.dsn)

        return self._engine