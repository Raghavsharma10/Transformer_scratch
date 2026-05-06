def similarity(self, other):
        """Get similarity as a discrete ratio (1.0 or 0.0)."""
        ratio = 1.0 if (str(self).lower() == str(other).lower()) else 0.0
        similarity = self.Similarity(ratio)
        return similarity