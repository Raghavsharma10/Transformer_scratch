def mask_nonfinite(self):
        """Extend the mask with the image elements where the intensity is NaN."""
        self.mask = np.logical_and(self.mask, (np.isfinite(self.intensity)))