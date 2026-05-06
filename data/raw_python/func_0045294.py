def textile(text, **kwargs):
    """
    Applies Textile conversion to a string, and returns the HTML.
    
    This is simply a pass-through to the ``textile`` template filter
    included in ``django.contrib.markup``, which works around issues
    PyTextile has with Unicode strings. If you're not using Django but
    want to use Textile with ``MarkupFormatter``, you'll need to
    supply your own Textile filter.
    
    """
    from django.contrib.markup.templatetags.markup import textile
    return textile(text)