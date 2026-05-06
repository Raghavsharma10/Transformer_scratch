def previous_song(self):
        """previous song for player to play

        NOTE: not the last played song
        """
        if self.current_song is None:
            return self._get_good_song(base=-1, direction=-1)

        if self.playback_mode == PlaybackMode.random:
            previous_song = self._get_good_song(direction=-1)
        else:
            current_index = self._songs.index(self.current_song)
            previous_song = self._get_good_song(base=current_index - 1, direction=-1)
        return previous_song