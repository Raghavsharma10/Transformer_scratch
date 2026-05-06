def register(model, admin=None, category=None):
    """ Decorator to registering you Admin class. """
    def _model_admin_wrapper(admin_class):

        site.register(model, admin_class=admin_class)

        if category:
            site.register_block(model, category)

        return admin_class
    return _model_admin_wrapper