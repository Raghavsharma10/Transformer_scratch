def css( self, filelist ):
        """This convenience function is only useful for html.
        It adds css stylesheet(s) to the document via the <link> element."""
      
        if isinstance( filelist, basestring ):
            self.link( href=filelist, rel='stylesheet', type='text/css', media='all' )
        else:
            for file in filelist:
                self.link( href=file, rel='stylesheet', type='text/css', media='all' )