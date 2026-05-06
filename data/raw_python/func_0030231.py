def dataframe(self,asql, logger = None):
        """Like query(), but returns a Pandas dataframe"""
        import pandas as pd
        from ambry.mprlib.exceptions import BadSQLError

        try:
            def yielder(cursor):

                for i, row in enumerate(cursor):
                    if i == 0:
                        yield [ e[0] for e in cursor.getdescription()]

                    yield row

            cursor = self.query(asql, logger)

            yld = yielder(cursor)

            header = next(yld)

            return pd.DataFrame(yld, columns=header)
        except BadSQLError as e:
            import traceback
            self._logger.error("SQL Error: {}".format( e))
            self._logger.debug(traceback.format_exc())