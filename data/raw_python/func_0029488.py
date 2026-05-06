def skip_branch(self, level):
        """ Abandon the branch at the provided level and skip to the next.

        When we call out to skip to the next branch of the search space, we
        push sublevel pieces to the maximum positions of the board. So that the
        next time the permutation iterator is called, it can produce the vector
        state of the next adjacent branch. See #3.
        """
        for i in range(level + 1, self.depth):
            self.indexes[i] = self.range_size - 1