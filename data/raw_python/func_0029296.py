def _get_create_update_kwargs(self, value, common_kw):
        """ Get kwargs common to create, update, replace. """
        kw = common_kw.copy()
        kw['body'] = value
        if '_self' in value:
            kw['headers'] = [('Location', value['_self'])]
        return kw