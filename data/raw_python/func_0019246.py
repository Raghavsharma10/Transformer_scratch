def open_files(subseqs):
        """Open file statements."""
        print('            . open_files')
        lines = Lines()
        lines.add(1, 'cpdef open_files(self, int idx):')
        for seq in subseqs:
            lines.add(2, 'if self._%s_diskflag:' % seq.name)
            lines.add(3, 'self._%s_file = fopen(str(self._%s_path).encode(), '
                         '"rb+")' % (2*(seq.name,)))
            if seq.NDIM == 0:
                lines.add(3,
                          'fseek(self._%s_file, idx*8, SEEK_SET)' % seq.name)
            else:
                lines.add(3, 'fseek(self._%s_file, idx*self._%s_length*8, '
                             'SEEK_SET)' % (2*(seq.name,)))
        return lines