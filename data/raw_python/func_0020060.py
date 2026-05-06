def unmasked(self, depth=0.01):
        """Return the unmasked overfitting metric for a given transit depth."""
        return 1 - (np.hstack(self._O2) +
                    np.hstack(self._O3) / depth) / np.hstack(self._O1)