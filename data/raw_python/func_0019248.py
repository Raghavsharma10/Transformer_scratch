def load_data(subseqs):
        """Load data statements."""
        print('            . load_data')
        lines = Lines()
        lines.add(1, 'cpdef inline void load_data(self, int idx) %s:' % _nogil)
        lines.add(2, 'cdef int jdx0, jdx1, jdx2, jdx3, jdx4, jdx5')
        for seq in subseqs:
            lines.add(2, 'if self._%s_diskflag:' % seq.name)
            if seq.NDIM == 0:
                lines.add(3, 'fread(&self.%s, 8, 1, self._%s_file)'
                             % (2*(seq.name,)))
            else:
                lines.add(3, 'fread(&self.%s[0], 8, self._%s_length, '
                             'self._%s_file)' % (3*(seq.name,)))
            lines.add(2, 'elif self._%s_ramflag:' % seq.name)
            if seq.NDIM == 0:
                lines.add(3, 'self.%s = self._%s_array[idx]' % (2*(seq.name,)))
            else:
                indexing = ''
                for idx in range(seq.NDIM):
                    lines.add(3+idx, 'for jdx%d in range(self._%s_length_%d):'
                                     % (idx, seq.name, idx))
                    indexing += 'jdx%d,' % idx
                indexing = indexing[:-1]
                lines.add(3+seq.NDIM, 'self.%s[%s] = self._%s_array[idx,%s]'
                                      % (2*(seq.name, indexing)))
        return lines