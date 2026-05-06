def get_locale_with_proxy(proxy):
    """Given a Proxy, returns the Locale

    This assumes that instantiating a dlkit.mongo.locale.objects.Locale
    without constructor arguments wlll return the default Locale.

    """
    from .locale.objects import Locale
    if proxy is not None:
            locale = proxy.get_locale()
            if locale is not None:
                return locale
    return Locale()