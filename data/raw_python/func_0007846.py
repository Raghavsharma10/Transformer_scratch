def score(self):
        """ Returns the sum of the accidental dignities
        score.
        
        """
        if not self.scoreProperties:
            self.scoreProperties = self.getScoreProperties()
        return sum(self.scoreProperties.values())