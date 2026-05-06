def modeldeclarations(self):
        """Attribute declarations of the model class."""
        lines = Lines()
        lines.add(0, '@cython.final')
        lines.add(0, 'cdef class Model(object):')
        lines.add(1, 'cdef public int idx_sim')
        lines.add(1, 'cdef public Parameters parameters')
        lines.add(1, 'cdef public Sequences sequences')
        if hasattr(self.model, 'numconsts'):
            lines.add(1, 'cdef public NumConsts numconsts')
        if hasattr(self.model, 'numvars'):
            lines.add(1, 'cdef public NumVars numvars')
        return lines