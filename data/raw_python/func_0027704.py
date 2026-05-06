def _getPowerupInterfaces(self):
        """
        Collect powerup interfaces this object declares that it can be
        installed on.
        """
        powerupInterfaces = getattr(self.__class__, "powerupInterfaces", ())
        pifs = []
        for x in powerupInterfaces:
            if isinstance(x, type(Interface)):
                #just an interface
                pifs.append((x, 0))
            else:
                #an interface and a priority
                pifs.append(x)

        m = getattr(self, "__getPowerupInterfaces__", None)
        if m is not None:
            pifs = m(pifs)
            try:
                pifs = [(i, p) for (i, p) in pifs]
            except ValueError:
                raise ValueError("return value from %r.__getPowerupInterfaces__"
                                 " not an iterable of 2-tuples" % (self,))
        return pifs