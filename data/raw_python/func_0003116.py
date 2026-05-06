def div( txt, *args, **kwargs ):
    """
    Create & return an HTML <div> element by wrapping the passed text buffer.
      @param txt (basestring): the text buffer to use
      @param *args (list): if present, \c txt is considered a Python format
        string, and the arguments are formatted into it
      @param kwargs (dict): the \c css field can contain the CSS class for the
        <div> element
    """
    if args:
        txt = txt.format( *args )
    css = kwargs.get('css',HTML_DIV_CLASS)
    return u'<div class="{}">{!s}</div>'.format( css, txt )