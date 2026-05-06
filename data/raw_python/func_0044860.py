def sound_speed(self, value):
        """Sets the sound speed in m/s. Default is 1525.0. If this function is
        called, `sound_speed_mode` will be set to fixed."""
        if not self.sound_speed_mode:
            self.sound_speed_mode = 1
        self.pdx.SoundSpeed = float(value)