def render_create(self, value, system, common_kw):
        """ Render response for view `create` method (collection POST) """
        kw = self._get_create_update_kwargs(value, common_kw)
        return JHTTPCreated(**kw)