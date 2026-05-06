def loudest_time(self, start=0, duration=0):
        """Find the loudest time in the window given by start and duration
        Returns frame number in context of entire track, not just the window.

        :param integer start: Start frame
        :param integer duration: Number of frames to consider from start
        :returns: Frame number of loudest frame
        :rtype: integer
        """
        if duration == 0:
            duration = self.sound.nframes
        self.current_frame = start
        arr = self.read_frames(duration)
        # get the frame of the maximum amplitude
        # different names for the same thing...
        # max_amp_sample = a.argmax(axis=0)[a.max(axis=0).argmax()]
        max_amp_sample = int(np.floor(arr.argmax()/2)) + start
        return max_amp_sample