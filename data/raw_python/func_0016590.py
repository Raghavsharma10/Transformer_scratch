def reset(self):
        """Reset state."""
        from samplerate.lowlevel import src_reset
        if self._state is None:
            self._create()
        src_reset(self._state)