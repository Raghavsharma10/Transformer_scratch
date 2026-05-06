def read(self, num_frames):
        """Read a number of frames from the resampler.

        Parameters
        ----------
        num_frames : int
            Number of frames to read.

        Returns
        -------
        output_data : ndarray
            Resampled frames as a (`num_output_frames`, `num_channels`) or
            (`num_output_frames`,) array. Note that this may return fewer frames
            than requested, for example when no more input is available.
        """
        from samplerate.lowlevel import src_callback_read, src_error
        from samplerate.exceptions import ResamplingError

        if self._state is None:
            self._create()
        if self._channels > 1:
            output_shape = (num_frames, self._channels)
        elif self._channels == 1:
            output_shape = (num_frames, )
        output_data = np.empty(output_shape, dtype=np.float32)

        ret = src_callback_read(self._state, self._ratio, num_frames,
                                output_data)
        if ret == 0:
            error = src_error(self._state)
            if error:
                raise ResamplingError(error)

        return (output_data[:ret, :]
                if self._channels > 1 else output_data[:ret])