def mean(self, only_valid=True) -> ErrorValue:
        """Calculate the mean of the pixels, not counting the masked ones if only_valid is True."""
        if not only_valid:
            intensity = self.intensity
            error = self.error
        else:
            intensity = self.intensity[self.mask]
            error = self.error[self.mask]
        return ErrorValue(intensity.mean(),
                          (error ** 2).mean() ** 0.5)