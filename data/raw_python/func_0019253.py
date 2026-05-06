def set_pointer1d(subseqs):
        """Set_pointer function for 1-dimensional link sequences."""
        print('            . set_pointer1d')
        lines = Lines()
        lines.add(1, 'cpdef inline set_pointer1d'
                     '(self, str name, pointerutils.PDouble value, int idx):')
        for seq in subseqs:
            lines.add(2, 'if name == "%s":' % seq.name)
            lines.add(3, 'self.%s[idx] = value.p_value' % seq.name)
        return lines