def update_deps( self ):
        """Update dependencies on all the fields on this Block instance."""
        klass = self.__class__

        for name in klass._fields:
            self.update_deps_on_field( name )
        return