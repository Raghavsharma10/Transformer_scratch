def similarity(self, other):
        """Get similarity as a ratio of the two numbers."""
        numerator, denominator = sorted((self.value, other.value))
        try:
            ratio = float(numerator) / denominator
        except ZeroDivisionError:
            ratio = 0.0 if numerator else 1.0
        similarity = self.Similarity(ratio)
        return similarity