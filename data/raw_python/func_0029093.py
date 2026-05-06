def info_hash_base32(self):
        """
        Returns the base32 info hash of the torrent. Useful for generating
        magnet links.

        .. note:: ``generate()`` must be called first.
        """
        if getattr(self, '_data', None):
            return b32encode(sha1(bencode(self._data['info'])).digest())
        else:
            raise exceptions.TorrentNotGeneratedException