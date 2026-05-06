def make_app(global_conf, full_stack=True, **app_conf):
    """
    Set depotexample up with the settings found in the PasteDeploy configuration
    file used.
    
    :param global_conf: The global settings for depotexample (those
        defined under the ``[DEFAULT]`` section).
    :type global_conf: dict
    :param full_stack: Should the whole TG2 stack be set up?
    :type full_stack: str or bool
    :return: The depotexample application with all the relevant middleware
        loaded.
    
    This is the PasteDeploy factory for the depotexample application.
    
    ``app_conf`` contains all the application-specific settings (those defined
    under ``[app:main]``.
    
   
    """
    app = make_base_app(global_conf, full_stack=True, **app_conf)
    
    # Wrap your base TurboGears 2 application with custom middleware here
    from depot.manager import DepotManager
    app = DepotManager.make_middleware(app)

    return app