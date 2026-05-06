def _get_values_from_auxiliaryfile(self, auxfile):
        """Try to return the parameter values from the auxiliary control file
        with the given name.

        Things are a little complicated here.  To understand this method, you
        should first take a look at the |parameterstep| function.
        """
        try:
            frame = inspect.currentframe().f_back.f_back
            while frame:
                namespace = frame.f_locals
                try:
                    subnamespace = {'model': namespace['model'],
                                    'focus': self}
                    break
                except KeyError:
                    frame = frame.f_back
            else:
                raise RuntimeError(
                    'Cannot determine the corresponding model.  Use the '
                    '`auxfile` keyword in usual parameter control files only.')
            filetools.ControlManager.read2dict(auxfile, subnamespace)
            try:
                subself = subnamespace[self.name]
            except KeyError:
                raise RuntimeError(
                    f'The selected file does not define value(s) for '
                    f'parameter {self.name}')
            return subself.values
        except BaseException:
            objecttools.augment_excmessage(
                f'While trying to extract information for parameter '
                f'`{self.name}` from file `{auxfile}`')