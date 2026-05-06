def n_frames(self, duration):
        '''Get the number of frames for a given duration

        Parameters
        ----------
        duration : number >= 0
            The duration, in seconds

        Returns
        -------
        n_frames : int >= 0
            The number of frames at this extractor's sampling rate and
            hop length
        '''

        return int(time_to_frames(duration, sr=self.sr,
                                  hop_length=self.hop_length))