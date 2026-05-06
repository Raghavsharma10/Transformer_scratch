def extract(self, node, condition, skip=0):
        """
        Extract a single node that matches the provided condition,
        otherwise a TypeError is raised.  An optional skip parameter can
        be provided to specify how many matching nodes are to be skipped
        over.
        """

        for child in self.filter(node, condition):
            if not skip:
                return child
            skip -= 1
        raise TypeError('no match found')