def cached_idxs(method):
    """ this function is used as a decorator for caching """
    def method_wrapper(self,*args,**kwargs):
        tail = '_'.join(str(idx) for idx in args)
        _cache_attr_name = '_cache_'+method.__name__+'_'+tail
        _bool_attr_name  = '_cached_'+method.__name__+'_'+tail
        is_cached = getattr(self,_bool_attr_name)
        if not is_cached:
            result = method(self, *args, **kwargs)
            setattr(self, _cache_attr_name, result)
            setattr(self, _bool_attr_name, True)
        return getattr(self,_cache_attr_name)
    return method_wrapper