def from_file(cls, f):
        """
        Constructs a :class:`~mwxml.iteration.dump.Dump` from a `file` pointer.

        :Parameters:
            f : `file`
                A plain text file pointer containing XML to process
        """
        element = ElementIterator.from_file(f)
        assert element.tag == "mediawiki"
        return cls.from_element(element)