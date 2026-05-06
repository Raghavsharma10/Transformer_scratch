def render_update_many(self, value, system, common_kw):
        """ Render response for view `update_many` method
        (collection PUT/PATCH)
        """
        msg = 'Updated {} {}(s) objects'.format(
            value, system['view'].Model.__name__)
        return JHTTPOk(msg, **common_kw.copy())