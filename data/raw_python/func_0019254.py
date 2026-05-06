def numericalparameters(self):
        """Numeric parameter declaration lines."""
        lines = Lines()
        if self.model.NUMERICAL:
            lines.add(0, '@cython.final')
            lines.add(0, 'cdef class NumConsts(object):')
            for name in ('nmb_methods', 'nmb_stages'):
                lines.add(1, 'cdef public %s %s' % (TYPE2STR[int], name))
            for name in ('dt_increase', 'dt_decrease'):
                lines.add(1, 'cdef public %s %s' % (TYPE2STR[float], name))
            lines.add(1, 'cdef public configutils.Config pub')
            lines.add(1, 'cdef public double[:, :, :] a_coefs')
            lines.add(0, 'cdef class NumVars(object):')
            for name in ('nmb_calls', 'idx_method', 'idx_stage'):
                lines.add(1, 'cdef public %s %s' % (TYPE2STR[int], name))
            for name in ('t0', 't1', 'dt', 'dt_est',
                         'error', 'last_error', 'extrapolated_error'):
                lines.add(1, 'cdef public %s %s' % (TYPE2STR[float], name))
            lines.add(1, 'cdef public %s f0_ready' % TYPE2STR[bool])
        return lines