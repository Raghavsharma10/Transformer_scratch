def merged_series(cls, *series, **kwargs):
        '''Merge ``series`` and return the results without storing data
in the backend server.'''
        router, backend = cls.check_router(None, *series)
        if backend:
            target = router.register(cls(), backend)
            router.session().add(target)
            target._merge(*series, **kwargs)
            backend = target.backend
            return backend.execute(
                backend.structure(target).irange_and_delete(),
                target.load_data)