def _register_admin(admin_site, model, admin_class):
    """ Register model in the admin, ignoring any previously registered models.
        Alternatively it could be used in the future to replace a previously 
        registered model.
    """
    try:
        admin_site.register(model, admin_class)
    except admin.sites.AlreadyRegistered:
        pass