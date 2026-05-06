def set_ipcsem_params(self, ftok=None, persistent=None):
        """Sets ipcsem lock engine params.

        :param str|unicode ftok: Set the ipcsem key via ftok() for avoiding duplicates.

        :param bool persistent: Do not remove ipcsem's on shutdown.

        """
        self._set('ftok', ftok)
        self._set('persistent-ipcsem', persistent, cast=bool)

        return self._section