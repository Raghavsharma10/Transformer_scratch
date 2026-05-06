def create_track(self, path_in_ipod=None, checksum=None):
        """
        :param path_in_ipod: the path of audio file in the iPod base
        :param checksum: CHECKSUM of the audio file in member audiodb
        :return: a new Track, you may want append it to the playlist.tracks
        """
        if bool(path_in_ipod) == bool(checksum):
            raise Exception

        if not path_in_ipod:
            path_in_ipod = self.audiodb.get_voice(checksum)

        track = Track(self, path_in_ipod=path_in_ipod)

        return track