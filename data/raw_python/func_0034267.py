def open( self, **kwargs ):
        """Append an opening tag."""

        if self.tag in self.parent.twotags or self.tag in self.parent.onetags:
            self.render( self.tag, False, None, kwargs )
        elif self.mode == 'strict_html' and self.tag in self.parent.deptags:
            raise DeprecationError( self.tag )