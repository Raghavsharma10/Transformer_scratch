def dict(self):
        """The dict representation of this sentence."""
        return {
            'raw': self.raw,
            'start_index': self.start_index,
            'end_index': self.end_index,
            'stripped': self.stripped,
            'noun_phrases': self.noun_phrases,
            'polarity': self.polarity,
            'subjectivity': self.subjectivity,
        }