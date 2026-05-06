def get_mode(self, old_mode=None):
        """Returns output mode. If `mode` not set it will try to guess best
        mode, or next best mode comparing to old mode

        """
        if self.mode is not None:
            return self.mode
        assert self.can_write, "This format does not have a supported output mode."
        if old_mode is None:
            return self.output_modes[0]
        if old_mode in self.output_modes:
            return old_mode
        # now let's get best mode available from supported
        try:
            idx = PILLOW_MODES.index(old_mode)
        except ValueError:
            # maybe some unknown or uncommon mode
            return self.output_modes[0]
        for mode in PILLOW_MODES[idx+1:]:
            if mode in self.output_modes:
                return mode
        # since there is no better one, lets' look for closest one in opposite direction
        opposite = PILLOW_MODES[:idx]
        opposite.reverse()
        for mode in opposite:
            if mode in self.output_modes:
                return mode