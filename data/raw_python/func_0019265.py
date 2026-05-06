def pyxlines(self):
        """Cython code lines.

        Assumptions:
          * Function shall be a method
          * Method shall be inlined
          * Method returns nothing
          * Method arguments are of type `int` (except self)
          * Local variables are generally of type `int` but of type `double`
            when their name starts with `d_`
        """
        lines = ['    '+line for line in self.cleanlines]
        lines[0] = lines[0].replace('def ', 'cpdef inline void ')
        lines[0] = lines[0].replace('):', ') %s:' % _nogil)
        for name in self.untypedarguments:
            lines[0] = lines[0].replace(', %s ' % name, ', int %s ' % name)
            lines[0] = lines[0].replace(', %s)' % name, ', int %s)' % name)
        for name in self.untypedinternalvarnames:
            if name.startswith('d_'):
                lines.insert(1, '        cdef double ' + name)
            else:
                lines.insert(1, '        cdef int ' + name)
        return Lines(*lines)