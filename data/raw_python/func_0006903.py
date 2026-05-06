def update_buffer_with_value( self, value, buffer, parent=None ):
        """Write a Python object into a byte array, using the field definition.

        value
            Input Python object to process.

        buffer
            Output byte array to encode value into.

        parent
            Parent block object where this Field is defined. Used for e.g.
            evaluating Refs.
        """
        assert common.is_bytes( buffer )
        self.validate( value, parent )
        return