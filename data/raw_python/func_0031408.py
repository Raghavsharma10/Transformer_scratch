def toDict(self):
        """
        Get information about a title alignment as a dictionary.

        @return: A C{dict} representation of the title aligment.
        """
        return {
            'hsps': [hsp.toDict() for hsp in self.hsps],
            'read': self.read.toDict(),
        }