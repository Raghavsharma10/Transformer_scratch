def expose(self, key=None):
        """
        Expose the decorated method for this L{Exposer} with the given key.  A
        method which is exposed will be able to be retrieved by this
        L{Exposer}'s C{get} method with that key.  If no key is provided, the
        key is the method name of the exposed method.

        Use like so::

            class MyClass:
                @someExposer.expose()
                def foo(): ...

        or::

            class MyClass:
                @someExposer.expose('foo')
                def unrelatedMethodName(): ...

        @param key: a hashable object, used by L{Exposer.get} to look up the
        decorated method later.  If None, the key is the exposed method's name.

        @return: a 1-argument callable which records its input as exposed, then
        returns it.
        """
        def decorator(function):
            rkey = key
            if rkey is None:
                if isinstance(function, FunctionType):
                    rkey = function.__name__
                else:
                    raise NameRequired()
            if rkey not in self._exposed:
                self._exposed[rkey] = []
            self._exposed[rkey].append(function)
            return function
        return decorator