def assignrepr(self, prefix: str) -> str:
        """Return a |repr| string with a prefixed assignment."""
        with objecttools.repr_.preserve_strings(True):
            with hydpy.pub.options.ellipsis(2, optional=True):
                with objecttools.assignrepr_tuple.always_bracketed(False):
                    classname = objecttools.classname(self)
                    blanks = ' ' * (len(prefix+classname) + 1)
                    nodestr = objecttools.assignrepr_tuple(
                        self.nodes.names, blanks+'nodes=', 70)
                    elementstr = objecttools.assignrepr_tuple(
                        self.elements.names, blanks + 'elements=', 70)
                    return (f'{prefix}{classname}("{self.name}",\n'
                            f'{nodestr},\n'
                            f'{elementstr})')