def full_url(url='', domain=None, protocol='https'):
    """
    Prepend protocol (default to https) and domain name (default from the Site framework) to an url
    """
    if domain is None:
        from django.contrib.sites.models import Site
        domain = Site.objects.get_current().domain
    return f'{protocol}://{domain}{url}'