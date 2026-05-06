def serialised( self ):
        """Tuple containing the contents of the Block."""
        klass = self.__class__
        return ((klass.__module__, klass.__name__), tuple( (name, field.serialise( self._field_data[name], parent=self ) ) for name, field in klass._fields.items()))