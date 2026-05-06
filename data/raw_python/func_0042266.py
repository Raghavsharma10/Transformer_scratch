def get_bundle(self, current_bundle, url_kwargs, context_kwargs):
        """
        Returns the bundle to get the alias view from.
        If 'self.bundle_attr' is set, that bundle that it points to
        will be returned, otherwise the current_bundle will be
        returned.
        """
        if self.bundle_attr:
            if self.bundle_attr == PARENT:
                return current_bundle.parent

            view, name = current_bundle.get_view_and_name(self.bundle_attr)
            return view

        return current_bundle