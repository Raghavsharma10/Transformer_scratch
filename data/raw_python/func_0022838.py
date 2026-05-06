def writeTuple(self, val, what):
        """ Writes a tuple of numbers (on one line).
        """
        # Limit to three values. so RGBA data drops the alpha channel
        # Format can handle up to 3 texcords
        val = val[:3]
        # Make string
        val = ' '.join([str(v) for v in val])
        # Write line
        self.writeLine('%s %s' % (what, val))