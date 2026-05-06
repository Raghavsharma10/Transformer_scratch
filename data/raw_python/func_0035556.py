def validate_settings(settings):
    """
    Raise an error in prod if we see any insecure settings.

    This used to warn during development but that was changed in
    71718bec324c2561da6cc3990c927ee87362f0f7
    """
    from django.core.exceptions import ImproperlyConfigured
    if settings.SECRET_KEY == '':
        msg = 'settings.SECRET_KEY cannot be blank! Check your local settings'
        if not settings.DEBUG:
            raise ImproperlyConfigured(msg)

    if getattr(settings, 'SESSION_COOKIE_SECURE', None) is None:
        msg = ('settings.SESSION_COOKIE_SECURE should be set to True; '
               'otherwise, your session ids can be intercepted over HTTP!')
        if not settings.DEBUG:
            raise ImproperlyConfigured(msg)

    hmac = getattr(settings, 'HMAC_KEYS', {})
    if not len(hmac.keys()):
        msg = 'settings.HMAC_KEYS cannot be empty! Check your local settings'
        if not settings.DEBUG:
            raise ImproperlyConfigured(msg)