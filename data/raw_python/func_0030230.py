def query(self, asql, logger=None):
        """
        Execute an ASQL file and return the result of the first SELECT statement.

        :param asql:
        :param logger:
        :return:
        """
        import sqlparse
        from ambry.mprlib.exceptions import BadSQLError
        from ambry.bundle.asql_parser import process_sql
        from ambry.orm.exc import NotFoundError

        if not logger:
            logger = self._library.logger

        rec = process_sql(asql, self._library)

        for drop in reversed(rec.drop):

            if drop:
                connection = self._backend._get_connection()
                cursor = self._backend.query(connection, drop, fetch=False)
                cursor.close()

        for vid in rec.materialize:
            logger.debug('Materialize {}'.format(vid))
            self.materialize(vid, logger=logger)

        for vid in rec.install:
            logger.debug('Install {}'.format(vid))

            self.install(vid, logger=logger)


        for statement in rec.statements:

            statement = statement.strip()

            logger.debug("Process statement: {}".format(statement[:60]))

            if statement.lower().startswith('create'):
                logger.debug('    Create {}'.format(statement))
                connection = self._backend._get_connection()
                cursor = self._backend.query(connection, statement, fetch=False)

                cursor.close()

            elif statement.lower().startswith('select'):
                logger.debug('Run query {}'.format(statement))
                connection = self._backend._get_connection()
                return self._backend.query(connection, statement, fetch=False)

        for table_or_vid, columns in rec.indexes:

            logger.debug('Index {}'.format(table_or_vid))

            try:
                self.index(table_or_vid, columns)
            except NotFoundError as e:
                # Comon when the index table in's a VID, so no partition can be found.

                logger.debug('Failed to index {}; {}'.format(vid, e))
            except Exception as e:
                logger.error('Failed to index {}; {}'.format(vid, e))

        # A fake cursor that can be closed and iterated
        class closable_iterable(object):
            def close(self):
                pass

            def __iter__(self):
                pass

        return closable_iterable()