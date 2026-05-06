def iter_disjunctions(self):
        """Returns a generator over 2-element lists of indexes which must
        be mutually exclusive in a solution (i.e. pairs of nodes which represent
        the same path in opposite directions.)"""
        for i in range(1, len(self.endpoints), 2):
            yield [i, self.get_disjoint(i)]