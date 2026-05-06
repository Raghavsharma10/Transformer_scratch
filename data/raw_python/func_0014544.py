def save(self, filename):
        """
        Save host keys into a file, in the format used by openssh.  The order of
        keys in the file will be preserved when possible (if these keys were
        loaded from a file originally).  The single exception is that combined
        lines will be split into individual key lines, which is arguably a bug.

        @param filename: name of the file to write
        @type filename: str

        @raise IOError: if there was an error writing the file

        @since: 1.6.1
        """
        f = open(filename, 'w')
        for e in self._entries:
            line = e.to_line()
            if line:
                f.write(line)
        f.close()