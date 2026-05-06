def assignrepr(self, prefix='') -> str:
        """Return a |repr| string with a prefixed assignment."""
        with objecttools.repr_.preserve_strings(True):
            with hydpy.pub.options.ellipsis(2, optional=True):
                prefix += '%s(' % objecttools.classname(self)
                repr_ = objecttools.assignrepr_values(
                    sorted(self.names), prefix, 70)
                return repr_ + ')'