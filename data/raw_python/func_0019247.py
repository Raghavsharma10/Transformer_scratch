def close_files(subseqs):
        """Close file statements."""
        print('            . close_files')
        lines = Lines()
        lines.add(1, 'cpdef inline close_files(self):')
        for seq in subseqs:
            lines.add(2, 'if self._%s_diskflag:' % seq.name)
            lines.add(3, 'fclose(self._%s_file)' % seq.name)
        return lines