def set_ratio(self, new_ratio):
        """Set a new conversion ratio immediately."""
        from samplerate.lowlevel import src_set_ratio
        return src_set_ratio(self._state, new_ratio)