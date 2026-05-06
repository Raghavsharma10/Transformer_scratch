def save(self, outfile):
        """Save the image data to a file or file-like object"""
        if isinstance(outfile, compat.string_type):
            outfile = open(outfile, 'wb')
        outfile.write(self._contents)