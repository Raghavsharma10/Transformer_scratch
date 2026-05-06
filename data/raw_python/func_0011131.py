def divides(self):
        """List of indices of divisions between the constituent chunks."""
        acc = [0]
        for s in self.chunks:
            acc.append(acc[-1] + len(s))
        return acc