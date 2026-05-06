def set( self, instance, value, **kwargs ):
        """Set an attribute on an object using the Ref path.

        instance
            The object instance to traverse.

        value
            The value to set.

        Throws AttributeError if allow_write is False.
        """
        if not self._allow_write:
            raise AttributeError( "can't set Ref directly, allow_write is disabled" )
        target = instance
        for attr in self._path[:-1]:
            target = getattr( target, attr )
        setattr( target, self._path[-1], value )
        return