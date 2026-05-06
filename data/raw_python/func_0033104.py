def _sqlalchemy_on_finish(self):
        """
        Closes the sqlalchemy transaction. Rolls back if an error occurred.
        """

        if hasattr(self, "_db_conns"):
            try:
                if self.get_status() >= 200 and self.get_status() <= 399:
                    for db_conn in self._db_conns.values():
                        db_conn.commit()
                else:
                    for db_conn in self._db_conns.values():
                        db_conn.rollback()
            except:
                tornado.log.app_log.warning("Error occurred during database transaction cleanup: %s", str(sys.exc_info()[0]))
                raise
            finally:
                for db_conn in self._db_conns.values():
                    try:
                        db_conn.close()
                    except:
                        tornado.log.app_log.warning("Error occurred when closing the database connection", exc_info=True)