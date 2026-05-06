def html_escape(s, encoding='utf-8', encoding_errors='strict'):
    """ Return the HTML-escaped version of an input. """
    return escape(make_unicode(s, encoding, encoding_errors), quote=True)