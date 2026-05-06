def mask_nan(self):
        """Extend the mask with the image elements where the intensity is NaN."""
        self.mask = np.logical_and(self.mask, ~(np.isnan(self.intensity)))