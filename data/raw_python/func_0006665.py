def get( self, instance, **kwargs ):
        """Return an attribute from an object using the Ref path.

        instance
            The object instance to traverse.
        """
        target = instance
        for attr in self._path:
            target = getattr( target, attr )
        return target