def pattern_for_view(self, view, action):
        """
        Returns the URL pattern for the passed in action.
        """
        # if this view knows how to define a URL pattern, call that
        if getattr(view, 'derive_url_pattern', None):
            return view.derive_url_pattern(self.path, action)

        # otherwise take our best guess
        else:
            return r'^%s/%s/$' % (self.path, action)