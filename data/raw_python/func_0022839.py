def writeFace(self, val, what='f'):
        """ Write the face info to the net line.
        """
        # OBJ counts from 1
        val = [v + 1 for v in val]
        # Make string
        if self._hasValues and self._hasNormals:
            val = ' '.join(['%i/%i/%i' % (v, v, v) for v in val])
        elif self._hasNormals:
            val = ' '.join(['%i//%i' % (v, v) for v in val])
        elif self._hasValues:
            val = ' '.join(['%i/%i' % (v, v) for v in val])
        else:
            val = ' '.join(['%i' % v for v in val])
        # Write line
        self.writeLine('%s %s' % (what, val))