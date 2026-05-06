def render_delete_many(self, value, system, common_kw):
        """ Render response for view `delete_many` method (collection DELETE)
        """
        if isinstance(value, dict):
            return JHTTPOk(extra=value)
        msg = 'Deleted {} {}(s) objects'.format(
            value, system['view'].Model.__name__)
        return JHTTPOk(msg, **common_kw.copy())