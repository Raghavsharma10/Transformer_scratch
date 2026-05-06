def _reload_maybe(self):
        """
        Reload the config if the config\
        model has been updated. This is called\
        once on every request by the middleware.\
        Should not be called directly.
        """
        ConfigModel = apps.get_model('djconfig.Config')

        data = dict(
            ConfigModel.objects
                .filter(key='_updated_at')
                .values_list('key', 'value'))

        if (not hasattr(self, '_updated_at') or
                self._updated_at != data.get('_updated_at')):
            self._reload()