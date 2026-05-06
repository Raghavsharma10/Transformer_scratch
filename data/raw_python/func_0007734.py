def auto_register_inlines(admin_site, metadata_class):
    """ This is a questionable function that automatically adds our metadata
        inline to all relevant models in the site. 
    """
    inline_class = get_inline(metadata_class)

    for model, admin_class_instance in admin_site._registry.items():
        _monkey_inline(model, admin_class_instance, metadata_class, inline_class, admin_site)

    # Monkey patch the register method to automatically add an inline for this site.
    # _with_inline() is a decorator that wraps the register function with the same injection code
    # used above (_monkey_inline).
    admin_site.register = _with_inline(admin_site.register, admin_site, metadata_class, inline_class)