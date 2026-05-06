def get_template_loader(self, subdir='templates'):
        '''App-specific function to get the current app's template loader'''
        if self.request is None:
            raise ValueError("this method can only be called after the view middleware is run. Check that `django_mako_plus.middleware` is in MIDDLEWARE.")
        dmp = apps.get_app_config('django_mako_plus')
        return dmp.engine.get_template_loader(self.app, subdir)