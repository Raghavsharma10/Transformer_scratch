def _get_attr_list(self, attr):
        """Return user's attribute/attributes"""
        a = self._attrs.get(attr)
        if not a:
            return []
        if type(a) is list:
            r = [i.decode('utf-8', 'ignore') for i in a]
        else:
            r = [a.decode('utf-8', 'ignore')]
        return r