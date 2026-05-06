def similarity(self, other):
        """Get similarity as a ratio of the stripped text."""
        logging.debug("comparing %r and %r...", self.stripped, other.stripped)
        ratio = SequenceMatcher(a=self.stripped, b=other.stripped).ratio()
        similarity = self.Similarity(ratio)
        return similarity