def with_result_cache(func):
        """
        Decorator specifically for is_active.  If self.result_cache is set to a {}
        the is_active results will be cached for each set of params.
        """
        def inner(self, *args, **kwargs):
            dic = self.result_cache
            cache_key = None
            if dic is not None:
                cache_key = (args, tuple(kwargs.items()))
                try:
                    result = dic.get(cache_key)
                except TypeError as e:  # not hashable
                    log.debug('Switchboard result cache not active for this "%s" check due to: %s within args: %s',
                              args[0], e, repr(cache_key)[:200])
                    cache_key = None
                else:
                    if result is not None:
                        return result
            result = func(self, *args, **kwargs)
            if cache_key is not None:
                dic[cache_key] = result
            return result
        return inner