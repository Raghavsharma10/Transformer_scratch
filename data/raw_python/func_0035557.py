def cached(method):
    """ this function is used as a decorator for caching """
    _cache_attr_name = '_cache_'+method.__name__
    _bool_attr_name  = '_cached_'+method.__name__
    def method_wrapper(self,*args,**kwargs):
        is_cached = getattr(self,_bool_attr_name)
        if not is_cached:
            result = method(self, *args, **kwargs)
            setattr(self, _cache_attr_name, result)
            setattr(self, _bool_attr_name, True)
        return getattr(self,'_cache_'+method.__name__)
    return method_wrapper