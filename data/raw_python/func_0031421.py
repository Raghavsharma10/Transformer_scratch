def toDict(self):
        """
        Get information about the titles alignments as a dictionary.

        @return: A C{dict} representation of the titles aligments.
        """
        return {
            'scoreClass': self.scoreClass.__name__,
            'titles': dict((title, titleAlignments.toDict())
                           for title, titleAlignments in self.items()),
        }