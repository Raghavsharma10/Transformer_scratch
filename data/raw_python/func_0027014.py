def cached_per_instance():
    """Decorator that adds caching to an instance method.

    The cached value is stored so that it gets garbage collected together with the instance.

    The cached values are not stored when the object is pickled.

    """

    def cache_fun(fun):
        argspec = inspect2.getfullargspec(fun)
        arg_names = argspec.args[1:] + argspec.kwonlyargs  # remove self
        kwargs_defaults = get_kwargs_defaults(argspec)
        cache = {}

        def cache_key(args, kwargs):
            return get_args_tuple(args, kwargs, arg_names, kwargs_defaults)

        def clear_cache(instance_key, ref):
            del cache[instance_key]

        @functools.wraps(fun)
        def new_fun(self, *args, **kwargs):
            instance_key = id(self)
            if instance_key not in cache:
                ref = weakref.ref(self, functools.partial(clear_cache, instance_key))
                cache[instance_key] = (ref, {})
            instance_cache = cache[instance_key][1]

            k = cache_key(args, kwargs)
            if k not in instance_cache:
                instance_cache[k] = fun(self, *args, **kwargs)
            return instance_cache[k]

        # just so unit tests can check that this is cleaned up correctly
        new_fun.__cached_per_instance_cache__ = cache
        return new_fun

    return cache_fun