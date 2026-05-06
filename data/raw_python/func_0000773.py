def reload_context(self, es_based, **kwargs):
        """ Reload `self.context` object into a DB or ES object.

        A reload is performed by getting the object ID from :kwargs: and then
        getting a context key item from the new instance of `self._factory`
        which is an ACL class used by the current view.

        Arguments:
            :es_based: Boolean. Whether to init ACL ac es-based or not. This
                affects the backend which will be queried - either DB or ES
            :kwargs: Kwargs that contain value for current resource 'id_name'
                key
        """
        from .acl import BaseACL
        key = self._get_context_key(**kwargs)
        kwargs = {'request': self.request}
        if issubclass(self._factory, BaseACL):
            kwargs['es_based'] = es_based

        acl = self._factory(**kwargs)
        if acl.item_model is None:
            acl.item_model = self.Model

        self.context = acl[key]