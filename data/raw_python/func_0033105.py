def _sqlalchemy_on_connection_close(self):
        """
        Rollsback and closes the active session, since the client disconnected before the request
        could be completed.
        """

        if hasattr(self, "_db_conns"):
            try:
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