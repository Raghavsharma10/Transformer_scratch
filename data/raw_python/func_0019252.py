def dealloc(subseqs):
        """Deallocate memory for 1-dimensional link sequences."""
        print('            . dealloc')
        lines = Lines()
        lines.add(1, 'cpdef inline dealloc(self):')
        for seq in subseqs:
            lines.add(2, 'PyMem_Free(self.%s)' % seq.name)
        return lines