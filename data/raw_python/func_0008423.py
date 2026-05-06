def group(self, index, chunked=False):
        """ Returns a list of Word objects that match the given group.
            With chunked=True, returns a list of Word + Chunk objects - see Match.constituents().
            A group consists of consecutive constraints wrapped in { }, e.g.,
            search("{JJ JJ} NN", Sentence(parse("big black cat"))).group(1) => big black.
        """
        if index < 0 or index > len(self.pattern.groups):
            raise IndexError("no such group")
        if index > 0 and index <= len(self.pattern.groups):
            g = self.pattern.groups[index-1]
        if index == 0:
            g = self.pattern.sequence
        if chunked is True:
            return Group(self, self.constituents(constraint=[self.pattern.sequence.index(x) for x in g]))
        return Group(self, [w for w in self.words if self.constraint(w) in g])