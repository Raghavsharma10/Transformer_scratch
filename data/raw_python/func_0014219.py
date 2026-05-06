def build_default_link(self):
        '''Called when 'link' is not defined in the settings'''
        attrs = {}
        attrs["rel"] = "stylesheet"
        attrs["href"] ="{}?{:x}".format(
            os.path.join(settings.STATIC_URL, self.filepath).replace(os.path.sep, '/'),
            self.version_id,
        )
        attrs.update(self.options['link_attrs'])
        attrs["data-context"] = self.provider_run.uid       # can't be overridden
        return '<link{} />'.format(flatatt(attrs))