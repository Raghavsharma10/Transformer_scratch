def sanitize_allow(value, args=''):
    '''
    Strip HTML tags other than provided tags and attributes.
    Example usage:

    {% load sanitizer %}
    {{ post.body|sanitize_allow:'a, strong, img; href, src'}}
    '''
    if isinstance(value, basestring):
        allowed_tags = []
        allowed_attributes = []
        allowed_styles = []
        
        args = args.strip().split(';')
        if len(args) > 0:
            allowed_tags = [tag.strip() for tag in args[0].split(',')]
        if len(args) > 1:
            allowed_attributes = [attr.strip() for attr in args[1].split(',')]
            
        value = bleach.clean(value, tags=allowed_tags,
                             attributes=allowed_attributes, strip=True)
    return value