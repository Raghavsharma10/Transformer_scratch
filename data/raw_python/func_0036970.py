def to_array(self, channels=2):
        """Generate the array of volume multipliers for the dynamic"""
        if self.fade_type == "linear":
            return np.linspace(self.in_volume, self.out_volume,
                self.duration * channels)\
                .reshape(self.duration, channels)
        elif self.fade_type == "exponential":
            if self.in_volume < self.out_volume:
                return (np.logspace(8, 1, self.duration * channels,
                    base=.5) * (
                        self.out_volume - self.in_volume) / 0.5 + 
                        self.in_volume).reshape(self.duration, channels)
            else:
                return (np.logspace(1, 8, self.duration * channels, base=.5
                    ) * (self.in_volume - self.out_volume) / 0.5 + 
                    self.out_volume).reshape(self.duration, channels)
        elif self.fade_type == "cosine":
            return