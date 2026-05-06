def clone_data( self, source ):
        """Clone data from another Block.

        source
            Block instance to copy from.
        """
        klass = self.__class__
        assert isinstance( source, klass )

        for name in klass._fields:
            self._field_data[name] = getattr( source, name )