def save(self, outfile):
        """Save the image data to a file or file-like object"""
        if isinstance(outfile, compat.string_type):
            outfile = open(outfile, 'wb')
        assert hasattr(outfile, 'write') and callable(outfile.write), \
            "Expect a file or file-like object with a .write() method"
        outfile.write(self.data)