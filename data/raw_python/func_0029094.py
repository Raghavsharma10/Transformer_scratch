def info_hash(self):
        """
        :return: The SHA-1 info hash of the torrent. Useful for generating
            magnet links.

        .. note:: ``generate()`` must be called first.
        """
        if getattr(self, '_data', None):
            return sha1(bencode(self._data['info'])).hexdigest()
        else:
            raise exceptions.TorrentNotGeneratedException