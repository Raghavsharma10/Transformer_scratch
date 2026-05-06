def sanitize(value):
    '''
    Sanitizes strings according to SANITIZER_ALLOWED_TAGS,
    SANITIZER_ALLOWED_ATTRIBUTES and SANITIZER_ALLOWED_STYLES variables in
    settings.

    Example usage:

    {% load sanitizer %}
    {{ post.content|escape_html }}

    '''
    if isinstance(value, basestring):
        value = bleach.clean(value, tags=ALLOWED_TAGS,
                             attributes=ALLOWED_ATTRIBUTES, 
                             styles=ALLOWED_STYLES, strip=False)
    return value