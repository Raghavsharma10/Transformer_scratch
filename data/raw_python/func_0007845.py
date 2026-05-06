def getActiveProperties(self):
        """ Returns the non-zero accidental dignities. """
        score = self.getScoreProperties()
        return {key: value for (key, value) in score.items()
                if value != 0}