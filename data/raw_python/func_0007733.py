def _with_inline(func, admin_site, metadata_class, inline_class):
    """ Decorator for register function that adds an appropriate inline."""   

    def register(model_or_iterable, admin_class=None, **options):
        # Call the (bound) function we were given.
        # We have to assume it will be bound to admin_site
        func(model_or_iterable, admin_class, **options)
        _monkey_inline(model_or_iterable, admin_site._registry[model_or_iterable], metadata_class, inline_class, admin_site)

    return register