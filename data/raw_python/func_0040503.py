def similarity(self, other):
        """Get similarity as a ratio of the two texts."""
        ratio = SequenceMatcher(a=self.value, b=other.value).ratio()
        similarity = self.Similarity(ratio)
        return similarity