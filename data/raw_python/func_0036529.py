def to_array(self, channels=2):
        """Return the array of multipliers for the dynamic"""
        if channels == 1:
            return self.volume_frames.reshape(-1, 1)
        if channels == 2:
            return np.tile(self.volume_frames, (2, 1)).T
        raise Exception(
            "RawVolume doesn't know what to do with %s channels" % channels)