def sequences(self):
        """Sequence declaration lines."""
        lines = Lines()
        lines.add(0, '@cython.final')
        lines.add(0, 'cdef class Sequences(object):')
        for subseqs in self.model.sequences:
            lines.add(1, 'cdef public %s %s'
                         % (objecttools.classname(subseqs), subseqs.name))
        if getattr(self.model.sequences, 'states', None) is not None:
            lines.add(1, 'cdef public StateSequences old_states')
            lines.add(1, 'cdef public StateSequences new_states')
        for subseqs in self.model.sequences:
            print('        - %s' % subseqs.name)
            lines.add(0, '@cython.final')
            lines.add(0, 'cdef class %s(object):'
                         % objecttools.classname(subseqs))
            for seq in subseqs:
                ctype = 'double' + NDIM2STR[seq.NDIM]
                if isinstance(subseqs, sequencetools.LinkSequences):
                    if seq.NDIM == 0:
                        lines.add(1, 'cdef double *%s' % seq.name)
                    elif seq.NDIM == 1:
                        lines.add(1, 'cdef double **%s' % seq.name)
                        lines.add(1, 'cdef public int len_%s' % seq.name)
                else:
                    lines.add(1, 'cdef public %s %s' % (ctype, seq.name))
                lines.add(1, 'cdef public int _%s_ndim' % seq.name)
                lines.add(1, 'cdef public int _%s_length' % seq.name)
                for idx in range(seq.NDIM):
                    lines.add(1, 'cdef public int _%s_length_%d'
                                 % (seq.name, idx))
                if seq.NUMERIC:
                    ctype_numeric = 'double' + NDIM2STR[seq.NDIM+1]
                    lines.add(1, 'cdef public %s _%s_points'
                                 % (ctype_numeric, seq.name))
                    lines.add(1, 'cdef public %s _%s_results'
                                 % (ctype_numeric, seq.name))
                    if isinstance(subseqs, sequencetools.FluxSequences):
                        lines.add(1, 'cdef public %s _%s_integrals'
                                     % (ctype_numeric, seq.name))
                        lines.add(1, 'cdef public %s _%s_sum'
                                     % (ctype, seq.name))
                if isinstance(subseqs, sequencetools.IOSequences):
                    lines.extend(self.iosequence(seq))
            if isinstance(subseqs, sequencetools.InputSequences):
                lines.extend(self.load_data(subseqs))
            if isinstance(subseqs, sequencetools.IOSequences):
                lines.extend(self.open_files(subseqs))
                lines.extend(self.close_files(subseqs))
                if not isinstance(subseqs, sequencetools.InputSequence):
                    lines.extend(self.save_data(subseqs))
            if isinstance(subseqs, sequencetools.LinkSequences):
                lines.extend(self.set_pointer(subseqs))
        return lines