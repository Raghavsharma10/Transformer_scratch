def set_pointer(self, subseqs):
        """Set_pointer functions for link sequences."""
        lines = Lines()
        for seq in subseqs:
            if seq.NDIM == 0:
                lines.extend(self.set_pointer0d(subseqs))
            break
        for seq in subseqs:
            if seq.NDIM == 1:
                lines.extend(self.alloc(subseqs))
                lines.extend(self.dealloc(subseqs))
                lines.extend(self.set_pointer1d(subseqs))
            break
        return lines