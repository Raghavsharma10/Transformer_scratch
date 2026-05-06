def increment(self):
        """ Increment the last permutation we returned to the next. """
        # Increment position from the deepest place of the tree first.
        for index in reversed(range(self.depth)):
            self.indexes[index] += 1
            # We haven't reached the end of board, no need to adjust upper
            # level.
            if self.indexes[index] < self.range_size:
                break
            # We've reached the end of board. Reset current level and increment
            # the upper level.
            self.indexes[index] = 0

        # Now that we incremented our indexes, we need to deduplicate positions
        # shering the same UIDs, by aligning piece's indexes to their parents.
        # This works thanks to the sort performed on self.pieces
        # initialization. See #7.
        for i in range(self.depth - 1):
            if (self.pieces[i] == self.pieces[i + 1]) and (
                    self.indexes[i] > self.indexes[i + 1]):
                self.indexes[i + 1] = self.indexes[i]