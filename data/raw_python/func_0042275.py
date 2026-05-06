def get_navigation(self, request, **kwargs):
        """
        Generates a list of tuples based on the values
        in `self.navigation` that are the side navigation links
        for this bundle. The tuple format is (url, title).
        """

        if self.navigation == self.parent_attr:
            if self.parent:
                return self.parent.get_navigation(request, **kwargs)
            return ()
        else:
            return self._nav_from_tuple(request, self.navigation,
                                **kwargs)