def restructuredtext(text, **kwargs):
    """
    Applies reStructuredText conversion to a string, and returns the
    HTML.
    
    """
    from docutils import core
    parts = core.publish_parts(source=text,
                               writer_name='html4css1',
                               **kwargs)
    return parts['fragment']