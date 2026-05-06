def render( self, tag, single, between, kwargs ):
        """Append the actual tags to content."""

        out = "<%s" % tag
        for key, value in list( kwargs.items( ) ):
            if value is not None:               # when value is None that means stuff like <... checked>
                key = key.strip('_')            # strip this so class_ will mean class, etc.
                if key == 'http_equiv':         # special cases, maybe change _ to - overall?
                    key = 'http-equiv'
                elif key == 'accept_charset':
                    key = 'accept-charset'
                out = "%s %s=\"%s\"" % ( out, key, escape( value ) )
            else:
                out = "%s %s" % ( out, key )
        if between is not None:
            out = "%s>%s</%s>" % ( out, between, tag )
        else:
            if single:
                out = "%s />" % out
            else:
                out = "%s>" % out
        if self.parent is not None:
            self.parent.content.append( out )
        else:
            return out