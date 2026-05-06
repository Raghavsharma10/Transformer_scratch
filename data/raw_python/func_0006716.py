def get_size( self ):
        """Get the projected size (in bytes) of the exported data from this Block instance."""
        klass = self.__class__
        size = 0
        for name in klass._fields:
            size = max( size, klass._fields[name].get_end_offset( self._field_data[name], parent=self ) )
        for check in klass._checks.values():
            size = max( size, check.get_end_offset( parent=self ) )
        return size