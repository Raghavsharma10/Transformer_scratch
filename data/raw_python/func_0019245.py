def iosequence(seq):
        """Special declaration lines for the given |IOSequence| object.
        """
        lines = Lines()
        lines.add(1, 'cdef public bint _%s_diskflag' % seq.name)
        lines.add(1, 'cdef public str _%s_path' % seq.name)
        lines.add(1, 'cdef FILE *_%s_file' % seq.name)
        lines.add(1, 'cdef public bint _%s_ramflag' % seq.name)
        ctype = 'double' + NDIM2STR[seq.NDIM+1]
        lines.add(1, 'cdef public %s _%s_array' % (ctype, seq.name))
        return lines