def get_frames(self, channels=2):
        """Get numpy array of frames corresponding to the segment.

        :param integer channels: Number of channels in output array
        :returns: Array of frames in the segment
        :rtype: numpy array

        """
        tmp_frame = self.track.current_frame
        self.track.current_frame = self.start
        frames = self.track.read_frames(self.duration, channels=channels)
        self.track.current_frame = tmp_frame

        for effect in self.effects:
            frames = effect.apply_to(frames, self.samplerate)

        return frames.copy()