def _fill_free_pool(self, override_min):
        """
        iterate over free connections and remove timeouted ones
        """
        while self.size < self.minsize:
            self._acquiring += 1
            try:
                conn = yield from connect(
                    database=self._database,
                    echo=self._echo,
                    loop=self._loop,
                    **self._conn_kwargs
                )
                self._free.append(conn)
                self._cond.notify()
            finally:
                self._acquiring -= 1
        if self._free:
            return

        if override_min and self.size < self.maxsize:
            self._acquiring += 1
            try:
                conn = yield from connect(
                    database=self._database,
                    echo=self._echo,
                    loop=self._loop,
                    **self._conn_kwargs
                )
                self._free.append(conn)
                self._cond.notify()
            finally:
                self._acquiring -= 1