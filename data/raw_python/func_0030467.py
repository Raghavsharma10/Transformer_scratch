def get(self, obj, key):
        """
        Retrieve 'key' from an instance of a class which previously exposed it.

        @param key: a hashable object, previously passed to L{Exposer.expose}.

        @return: the object which was exposed with the given name on obj's key.

        @raise MethodNotExposed: when the key in question was not exposed with
        this exposer.
        """
        if key not in self._exposed:
            raise MethodNotExposed()
        rightFuncs = self._exposed[key]
        T = obj.__class__
        seen = {}
        for subT in inspect.getmro(T):
            for name, value in subT.__dict__.items():
                for rightFunc in rightFuncs:
                    if value is rightFunc:
                        if name in seen:
                            raise MethodNotExposed()
                        return value.__get__(obj, T)
                seen[name] = True
        raise MethodNotExposed()