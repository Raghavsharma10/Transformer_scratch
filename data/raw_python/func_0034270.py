def metainfo( self, mydict ):
        """This convenience function is only useful for html.
        It adds meta information via the <meta> element, the argument is
        a dictionary of the form { 'name':'content' }."""

        if isinstance( mydict, dict ):
            for name, content in list( mydict.items( ) ):
                self.meta( name=name, content=content )
        else:
            raise TypeError( "Metainfo should be called with a dictionary argument of name:content pairs." )