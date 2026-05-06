def save(self) -> sql.Executant:
        """Prepare a SQL request to save the current modifications.
        Returns actually a LIST of requests (which may be of length one).
        Note than it can include modifications on other part of the data.
        After succes, the base should be updated.
        """
        r = self._dict_to_SQL(self.modifications)
        self.modifications.clear()
        return r