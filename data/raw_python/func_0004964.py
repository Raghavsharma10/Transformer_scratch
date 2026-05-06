def mask_negative(self):
        """Extend the mask with the image elements where the intensity is negative."""
        self.mask = np.logical_and(self.mask, ~(self.intensity < 0))