def getLoader(*a, **kw):
    """
    Deprecated.  Don't use this.
    """
    warn("xmantissa.publicweb.getLoader is deprecated, use "
         "PrivateApplication.getDocFactory or SiteTemplateResolver."
         "getDocFactory.", category=DeprecationWarning, stacklevel=2)
    from xmantissa.webtheme import getLoader
    return getLoader(*a, **kw)