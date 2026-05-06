def classify(self, text):
        """
        Chooses the highest scoring category for a sample of text

        :param text: sample text to classify
        :type text: str
        :return: the "winning" category
        :rtype: str
        """
        score = self.score(text)
        if not score:
            return None
        return sorted(score.items(), key=lambda v: v[1])[-1][0]