def initialise_shopify_session():
    """
    Initialise the Shopify session with the Shopify App's API credentials.
    """
    if not settings.SHOPIFY_APP_API_KEY or not settings.SHOPIFY_APP_API_SECRET:
        raise ImproperlyConfigured("SHOPIFY_APP_API_KEY and SHOPIFY_APP_API_SECRET must be set in settings")
    shopify.Session.setup(api_key=settings.SHOPIFY_APP_API_KEY, secret=settings.SHOPIFY_APP_API_SECRET)