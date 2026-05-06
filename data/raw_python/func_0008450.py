def word_counts(self):
        """Dictionary of word frequencies in this text."""
        counts = defaultdict(int)
        stripped_words = [lowerstrip(word) for word in self.words]
        for word in stripped_words:
            counts[word] += 1
        return counts