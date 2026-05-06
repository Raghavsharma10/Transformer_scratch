def lazy_val(func, with_del_hook=False):
    '''A memoize decorator for class properties.

    Return a cached property that is calculated by function `func` on first
    access.
    '''

    def hook_for(that):
        try:
            orig_del = that.__del__
        except AttributeError:
            orig_del = None

        def del_hook(*args, **kwargs):
            del that._cache[id(that)]
            del that._del_hook_cache[id(that)]
            if orig_del is not None:
                orig_del(that, *args, **kwargs)

        try:
            if orig_del is not None:
                that.__del__ = del_hook
        except AttributeError:
            # that.__del__ is a class property and cannot be changed by instance
            orig_del = None
        return del_hook

    def add_to_del_hook_cache(that):
        if with_del_hook:
            try:
                that._del_hook_cache[id(that)] = hook_for(that)
            except AttributeError:
                # when that._del_hook_cache not exists, it means it is not a
                # class property.  Then, we don't need a del_hook().
                pass

    @functools.wraps(func)
    def get(self):
        try:
            return self._cache[id(self)][func]
        except AttributeError:
            self._cache = {id(self): {}, }
            add_to_del_hook_cache(self)
        except KeyError:
            try:
                self._cache[id(self)]
            except KeyError:
                self._cache[id(self)] = {}
                add_to_del_hook_cache(self)
        val = self._cache[id(self)][func] = func(self)
        return val

    return property(get)