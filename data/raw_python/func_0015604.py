def strip_filter(value):
    '''
    Strips HTML tags from strings according to SANITIZER_ALLOWED_TAGS,
    SANITIZER_ALLOWED_ATTRIBUTES and SANITIZER_ALLOWED_STYLES variables in
    settings.

    Example usage:

    {% load sanitizer %}
    {{ post.content|strip_html }}

    '''
    if isinstance(value, basestring):
        value = bleach.clean(value, tags=ALLOWED_TAGS,
                             attributes=ALLOWED_ATTRIBUTES, 
                             styles=ALLOWED_STYLES, strip=True)
    return value