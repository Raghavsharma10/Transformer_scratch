def to_array(self, channels=2):
        """Generate the array of multipliers for the dynamic"""
        return np.linspace(self.volume, self.volume,
            self.duration * channels).reshape(self.duration, channels)