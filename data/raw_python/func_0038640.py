def count(self, key):
        """Return the number of pairs with key *key*."""
        count = 0
        pos = self.index(key, -1)
        if pos == -1:
            return count
        count += 1
        for i in range(pos+1, len(self)):
            if self[i][0] != key:
                break
            count += 1
        return count