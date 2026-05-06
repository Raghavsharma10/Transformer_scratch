def get_client_nowait(self):
        """Gets a Client object (not necessary connected).

        If max_size is reached, this method will return None (and won't block).

        Returns:
            A Client instance (not necessary connected) as result (or None).
        """
        if self.__sem is not None:
            if self.__sem._value == 0:
                return None
            self.__sem.acquire()
        _, client = self._get_client_from_pool_or_make_it()
        return client