def range_as_mono(self, start_sample, end_sample):
        """Get a range of frames as 1 combined channel

        :param integer start_sample: First frame in range
        :param integer end_sample: Last frame in range (exclusive)
        :returns: Track frames in range as 1 combined channel
        :rtype: 1d numpy array of length ``end_sample - start_sample``
        """
        tmp_current = self.current_frame
        self.current_frame = start_sample
        tmp_frames = self.read_frames(end_sample - start_sample)
        if self.channels == 2:
            frames = np.mean(tmp_frames, axis=1)
        elif self.channels == 1:
            frames = tmp_frames
        else:
            raise IOError("Input audio must have either 1 or 2 channels")
        self.current_frame = tmp_current
        return frames