def render_update(self, value, system, common_kw):
        """ Render response for view `update` method (item PATCH) """
        kw = self._get_create_update_kwargs(value, common_kw)
        return JHTTPOk('Updated', **kw)