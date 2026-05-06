def load(self, f, line=None):
        """Load this section from a file-like object"""
        if line is None:
            # in case the file contains only a fragment of an input file,
            # this is useful.
            line = f.readlin()
        words = line[1:].split()
        self.__name = words[0].upper()
        self.section_parameters = " ".join(words[1:])
        try:
            self.load_children(f)
        except EOFError:
            raise FileFormatError("Unexpected end of file, section '%s' not ended." % self.__name)