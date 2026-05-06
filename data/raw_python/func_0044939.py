def set_site():
    """
    This method is part of the prepare_data helper.
    Sets the site. Default implementation uses localhost.
    For production settings refine this helper.
    :return:
    """
    from django.contrib.sites.models import Site
    from django.conf import settings
    # Initially set localhost as default domain
    #
    site = Site.objects.get(id=settings.SITE_ID)
    site.domain = 'http://localhost:8000'
    site.name = 'http://localhost:8000'
    site.save()