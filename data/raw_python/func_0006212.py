def get_crumbs(self):
        """
        Get crumbs for navigation links.

        Returns:
            tuple:
                concatenated list of crumbs using these crumbs and the
                crumbs of the parent classes through ``__mro__``.
        """
        crumbs = []
        for cls in reversed(type(self).__mro__[1:]):
            crumbs.extend(getattr(cls, 'crumbs', ()))
        crumbs.extend(list(self.crumbs))
        return tuple(crumbs)