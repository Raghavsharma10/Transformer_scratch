def set_pointer0d(subseqs):
        """Set_pointer function for 0-dimensional link sequences."""
        print('            . set_pointer0d')
        lines = Lines()
        lines.add(1, 'cpdef inline set_pointer0d'
                     '(self, str name, pointerutils.PDouble value):')
        for seq in subseqs:
            lines.add(2, 'if name == "%s":' % seq.name)
            lines.add(3, 'self.%s = value.p_value' % seq.name)
        return lines