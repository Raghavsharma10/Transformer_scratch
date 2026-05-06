def toDict(self):
        """
        Get information about this read in a dictionary.

        @return: A C{dict} with keys/values for the attributes of self.
        """
        if six.PY3:
            result = super().toDict()
        else:
            result = AARead.toDict(self)

        result.update({
            'start': self.start,
            'stop': self.stop,
            'openLeft': self.openLeft,
            'openRight': self.openRight,
        })

        return result