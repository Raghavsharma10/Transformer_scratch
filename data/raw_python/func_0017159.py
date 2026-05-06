def remove_label(self, name):
        """Removes label ``name`` from this issue.

        :param str name: (required), name of the label to remove
        :returns: bool
        """
        url = self._build_url('labels', name, base_url=self._api)
        # Docs say it should be a list of strings returned, practice says it
        # is just a 204/404 response. I'm tenatively changing this until I
        # hear back from Support.
        return self._boolean(self._delete(url), 204, 404)