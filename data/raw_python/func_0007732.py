def _monkey_inline(model, admin_class_instance, metadata_class, inline_class, admin_site):
    """ Monkey patch the inline onto the given admin_class instance. """
    if model in metadata_class._meta.seo_models:
        # *Not* adding to the class attribute "inlines", as this will affect
        # all instances from this class. Explicitly adding to instance attribute.
        admin_class_instance.__dict__['inlines'] = admin_class_instance.inlines + [inline_class]

        # Because we've missed the registration, we need to perform actions
        # that were done then (on admin class instantiation)
        inline_instance = inline_class(admin_class_instance.model, admin_site)
        admin_class_instance.inline_instances.append(inline_instance)