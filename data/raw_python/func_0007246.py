def is_contiguous(self):
        '''Return whether entire collection is contiguous.'''
        previous = None
        for index in self.indexes:
            if previous is None:
                previous = index
                continue

            if index != (previous + 1):
                return False

            previous = index

        return True