def derive_single_object_url_pattern(slug_url_kwarg, path, action):
    """
    Utility function called by class methods for single object views
    """
    if slug_url_kwarg:
        return r'^%s/%s/(?P<%s>[^/]+)/$' % (path, action, slug_url_kwarg)
    else:
        return r'^%s/%s/(?P<pk>\d+)/$' % (path, action)