def get_cache_item(self):
        '''Gets the cached item. Raises AttributeError if it hasn't been set.'''
        if settings.DEBUG:
            raise AttributeError('Caching disabled in DEBUG mode')
        return getattr(self.template, self.options['template_cache_key'])