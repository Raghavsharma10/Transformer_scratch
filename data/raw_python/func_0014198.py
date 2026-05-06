def get_template_loader(self, app, subdir='templates', create=False):
        '''
        Returns a template loader object for the given app name in the given subdir.
        For example, get_template_loader('homepage', 'styles') will return
        a loader for the styles/ directory in the homepage app.

        The app parameter can be either an app name or an AppConfig instance.
        The subdir parameter is normally 'templates', 'scripts', or 'styles',
        but it can be any subdirectory name of the given app.

        Normally, you should not have to call this method.  Django automatically
        generates two shortcut functions for every DMP-registered apps,
        and these shortcut functions are the preferred way to render templates.

        This method is useful when you want a custom template loader to a directory
        that does not conform to the app_dir/templates/* pattern.

        If the loader is not found in the DMP cache, one of two things occur:
          1. If create=True, it is created automatically and returned.  This overrides
             the need to register the app as a DMP app.
          2. If create=False, a TemplateDoesNotExist is raised.  This is the default
             behavior.
        '''
        # ensure we have an AppConfig
        if app is None:
            raise TemplateDoesNotExist("Cannot locate loader when app is None")
        if not isinstance(app, AppConfig):
            app = apps.get_app_config(app)

        # get the loader with the path of this app+subdir
        path = os.path.join(app.path, subdir)

        # if create=False, the loader must already exist in the cache
        if not create:
            dmp = apps.get_app_config('django_mako_plus')
            if not dmp.is_registered_app(app):
                raise ValueError("{} is not registered with DMP [hint: check urls.py for include('django_mako_plus.urls')].".format(app))

        # return the template by path
        return self.get_template_loader_for_path(path, use_cache=True)