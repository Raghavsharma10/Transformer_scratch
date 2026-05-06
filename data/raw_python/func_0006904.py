def get_end_offset( self, value, parent=None, index=None ):
        """Return the end offset of the Field's data. Useful for chainloading.

        value
            Input Python object to process.

        parent
            Parent block object where this Field is defined. Used for e.g.
            evaluating Refs.

        index
            Index of the Python object to measure from. Used if the Field
            takes a list of objects.
        """
        return self.get_start_offset( value, parent, index ) + self.get_size( value, parent, index )