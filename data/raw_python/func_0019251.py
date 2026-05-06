def alloc(subseqs):
        """Allocate memory for 1-dimensional link sequences."""
        print('            . setlength')
        lines = Lines()
        lines.add(1, 'cpdef inline alloc(self, name, int length):')
        for seq in subseqs:
            lines.add(2, 'if name == "%s":' % seq.name)
            lines.add(3, 'self._%s_length_0 = length' % seq.name)
            lines.add(3, 'self.%s = <double**> '
                         'PyMem_Malloc(length * sizeof(double*))' % seq.name)
        return lines