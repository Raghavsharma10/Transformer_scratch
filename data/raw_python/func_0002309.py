def get_dummy_request(language=None):
    """
    Returns a Request instance populated with cms specific attributes.
    """

    if settings.ALLOWED_HOSTS and settings.ALLOWED_HOSTS != "*":
        host = settings.ALLOWED_HOSTS[0]
    else:
        host = Site.objects.get_current().domain

    request = RequestFactory().get("/", HTTP_HOST=host)
    request.session = {}
    request.LANGUAGE_CODE = language or settings.LANGUAGE_CODE
    # Needed for plugin rendering.
    request.current_page = None

    if 'django.contrib.auth' in settings.INSTALLED_APPS:
        from django.contrib.auth.models import AnonymousUser
        request.user = AnonymousUser()

    return request