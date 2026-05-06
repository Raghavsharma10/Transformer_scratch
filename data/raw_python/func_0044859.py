def sound_speed_mode(self, mode):
        """Sets sound speed mode; 0 or "measured" for measured; 1 or "fixed"
        for fixed."""
        if mode == "measured":
            mode = 0
        if mode == "fixed":
            mode = 1
        self.pdx.SoundSpeedMode = mode