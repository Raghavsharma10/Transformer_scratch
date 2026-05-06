def unescape( text ):
    """Inverse of escape."""
    
    if isinstance( text, basestring ):
        if '&amp;' in text:
            text = text.replace( '&amp;', '&' )
        if '&gt;' in text:
            text = text.replace( '&gt;', '>' )
        if '&lt;' in text:
            text = text.replace( '&lt;', '<' )
        if '&quot;' in text:
            text = text.replace( '&quot;', '\"' )

    return text