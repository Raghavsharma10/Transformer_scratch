def get_urls(self):
        """
        Returns urls handling bundles and views.
        This processes the 'item view' first in order
        and then adds any non item views at the end.
        """
        parts = []
        seen = set()

        # Process item views in order
        for v in list(self._meta.item_views)+list(self._meta.action_views):
            if not v in seen:
                view, name = self.get_view_and_name(v)
                if view and name:
                    parts.append(self.get_url(name, view, v))
                seen.add(v)

        # Process everything else that we have not seen
        for v in set(self._views).difference(seen):
            # Get the url name
            view, name = self.get_view_and_name(v)
            if view and name:
                parts.append(self.get_url(name, view, v))

        return parts