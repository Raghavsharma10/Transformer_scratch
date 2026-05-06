def _create(self):
        """Create new callback resampler."""
        from samplerate.lowlevel import ffi, src_callback_new, src_delete
        from samplerate.exceptions import ResamplingError

        state, handle, error = src_callback_new(
            self._callback, self._converter_type.value, self._channels)
        if error != 0:
            raise ResamplingError(error)
        self._state = ffi.gc(state, src_delete)
        self._handle = handle