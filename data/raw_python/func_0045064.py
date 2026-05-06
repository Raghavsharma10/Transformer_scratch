def smartypants(value):
    """
    Applies SmartyPants to a piece of text, applying typographic
    niceties.
    
    Requires the Python SmartyPants library to be installed; see
    http://web.chad.org/projects/smartypants.py/
    
    """
    try:
        from smartypants import smartyPants
    except ImportError:
        if settings.DEBUG:
            raise TemplateSyntaxError("Error in smartypants filter: the Python smartypants module is not installed or could not be imported")
        return value
    else:
        return smartyPants(value)