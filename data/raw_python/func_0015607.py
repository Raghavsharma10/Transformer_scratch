def strip_html(value, allowed_tags=[], allowed_attributes=[],
               allowed_styles=[]):
    """
    Template tag to strip html from string values. It accepts lists of
    allowed tags, attributes or stylesin comma separated string or list format.

    For example:

    {% load sanitizer %}
    {% strip_html '<a href="">bar</a> <script>alert('baz')</script>' "a,img' 'href',src' %}

    Will output:

    <a href="">bar</a> alert('baz');

    On django 1.4 you could also use keyword arguments:

    {% strip_html '<a href="">bar</a>' allowed_tags="a,img' allowed_attributes='href',src' %}    

    """
    if isinstance(value, basestring):
        value = bleach.clean(value, tags=allowed_tags,
                             attributes=allowed_attributes, 
                             styles=allowed_styles, strip=True)
    return value