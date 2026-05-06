def get_osdb_hash(self):
        """
        Get the hash of this local videofile
        :return: hash as string
        """
        if self._osdb_hash is None:
            self._osdb_hash = self._calculate_osdb_hash()
        return self._osdb_hash