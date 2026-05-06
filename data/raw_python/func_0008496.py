def annotate(self, word, pos=None, polarity=0.0, subjectivity=0.0, intensity=1.0, label=None):
        """ Annotates the given word with polarity, subjectivity and intensity scores,
            and optionally a semantic label (e.g., MOOD for emoticons, IRONY for "(!)").
        """
        w = self.setdefault(word, {})
        w[pos] = w[None] = (polarity, subjectivity, intensity)
        if label:
            self.labeler[word] = label