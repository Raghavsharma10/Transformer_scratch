def close( self ):
        """Append a closing tag unless element has only opening tag."""

        if self.tag in self.parent.twotags:
            self.parent.content.append( "</%s>" % self.tag )
        elif self.tag in self.parent.onetags:
            raise ClosingError( self.tag )
        elif self.parent.mode == 'strict_html' and self.tag in self.parent.deptags:
            raise DeprecationError( self.tag )