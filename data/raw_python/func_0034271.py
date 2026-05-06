def scripts( self, mydict ):
        """Only useful in html, mydict is dictionary of src:type pairs or a list
        of script sources [ 'src1', 'src2', ... ] in which case 'javascript' is assumed for type.
        Will be rendered as <script type='text/type' src=src></script>"""

        if isinstance( mydict, dict ):
            for src, type in list( mydict.items( ) ):
                self.script( '', src=src, type='text/%s' % type )
        else:
            try:
                for src in mydict:
                    self.script( '', src=src, type='text/javascript' )
            except:
                raise TypeError( "Script should be given a dictionary of src:type pairs or a list of javascript src's." )