def song(self):
        """the song associated with the project"""
        if self._song is None:
            self._song = Song(self._song_data)

        return self._song