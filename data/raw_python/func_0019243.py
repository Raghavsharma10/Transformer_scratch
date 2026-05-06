def parameters(self):
        """Parameter declaration lines."""
        lines = Lines()
        lines.add(0, '@cython.final')
        lines.add(0, 'cdef class Parameters(object):')
        for subpars in self.model.parameters:
            if subpars:
                lines.add(1, 'cdef public %s %s'
                             % (objecttools.classname(subpars), subpars.name))
        for subpars in self.model.parameters:
            if subpars:
                print('        - %s' % subpars.name)
                lines.add(0, '@cython.final')
                lines.add(0, 'cdef class %s(object):'
                             % objecttools.classname(subpars))
                for par in subpars:
                    try:
                        ctype = TYPE2STR[par.TYPE] + NDIM2STR[par.NDIM]
                    except KeyError:
                        ctype = par.TYPE + NDIM2STR[par.NDIM]
                    lines.add(1, 'cdef public %s %s' % (ctype, par.name))
        return lines