def replace(self, child, replacement):
        """Replace a child chunk with something else."""
        for i in range(len(self.chunks)):
            if self.chunks[i] == child:
                self.chunks[i] = replacement