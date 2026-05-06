def synchronized(cls, obj=None):
        """ synchronize on obj if obj is supplied.

        :param obj: the obj to lock on.  if none, lock to the function
        :return: return of the func.
        """

        def get_key(f, o):
            if o is None:
                key = hash(f)
            else:
                key = hash(o)
            return key

        def get_lock(f, o):
            key = get_key(f, o)
            if key not in cls.lock_map:
                with cls.lock_map_lock:
                    if key not in cls.lock_map:
                        cls.lock_map[key] = _init_lock()
            return cls.lock_map[key]

        def wrap(f):
            @functools.wraps(f)
            def new_func(*args, **kw):
                with get_lock(f, obj):
                    return f(*args, **kw)

            return new_func

        return wrap