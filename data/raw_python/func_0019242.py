def constants(self):
        """Constants declaration lines."""
        lines = Lines()
        for (name, member) in vars(self.cythonizer).items():
            if (name.isupper() and
                    (not inspect.isclass(member)) and
                    (type(member) in TYPE2STR)):
                ndim = numpy.array(member).ndim
                ctype = TYPE2STR[type(member)] + NDIM2STR[ndim]
                lines.add(0, 'cdef public %s %s = %s'
                             % (ctype, name, member))
        return lines