def item_options(self, **kwargs):
        """ Handle collection OPTIONS request.

        Singular route requests are handled a bit differently because
        singular views may handle POST requests despite being registered
        as item routes.
        """
        actions = self._item_actions.copy()
        if self._resource.is_singular:
            actions['create'] = ('POST',)
        methods = self._get_handled_methods(actions)
        return self._set_options_headers(methods)