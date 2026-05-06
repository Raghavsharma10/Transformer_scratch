def export_data( self ):
        """Export data to a byte array."""
        klass = self.__class__

        output = bytearray( b'\x00'*self.get_size() )

        # prevalidate all data before export.
        # this is important to ensure that any dependent fields
        # are updated beforehand, e.g. a count referenced
        # in a BlockField
        queue = []
        for name in klass._fields:
            self.scrub_field( name )
            self.validate_field( name )

        self.update_deps()

        for name in klass._fields:
            klass._fields[name].update_buffer_with_value(
                self._field_data[name], output, parent=self
            )

        for name, check in klass._checks.items():
            check.update_buffer( output, parent=self )
        return output