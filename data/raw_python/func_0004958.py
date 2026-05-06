def sum(self, only_valid=True) -> ErrorValue:
        """Calculate the sum of pixels, not counting the masked ones if only_valid is True."""
        if not only_valid:
            mask = 1
        else:
            mask = self.mask
        return ErrorValue((self.intensity * mask).sum(),
                          ((self.error * mask) ** 2).sum() ** 0.5)