def release(self, connection):
        """ Release a connection back into the pool.
        This method is thread safe.
        """
        with self.lock:
            connection.in_use = False
            self.cond.notify_all()