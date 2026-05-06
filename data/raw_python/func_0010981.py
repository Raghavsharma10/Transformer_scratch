def get_user_cache_key(**kwargs):
    """ Generate suitable key to cache twitter tag context
    """
    key = 'get_tweets_%s' % ('_'.join([str(kwargs[key]) for key in sorted(kwargs) if kwargs[key]]))
    not_allowed = re.compile('[^%s]' % ''.join([chr(i) for i in range(33, 128)]))
    key = not_allowed.sub('', key)
    return key