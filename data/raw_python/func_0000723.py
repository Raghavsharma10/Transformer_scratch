def connection(self):
        """
        A context manager that returns a connection
        to the server using some *session*.
        """
        conn = self.session(**self.options)
        try:
            for item in self.middlewares:
                item(conn)
            yield conn
        finally:
            conn.teardown()