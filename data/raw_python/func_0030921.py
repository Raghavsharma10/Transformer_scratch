def toDict(self):
        """
        Get information about the HSP as a dictionary.

        @return: A C{dict} representation of the HSP.
        """
        result = _Base.toDict(self)
        result['score'] = self.score.score
        return result