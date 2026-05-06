def remove(self, child):
        """Remove a child element."""
        for i in range(len(self)):
            if self[i] == child:
                del self[i]