def validate_url(value):
    """ Validate url. """
    if not re.match(VIMEO_URL_RE, value) and not re.match(YOUTUBE_URL_RE, value):
        raise ValidationError('Invalid URL - only Youtube, Vimeo can be used.')