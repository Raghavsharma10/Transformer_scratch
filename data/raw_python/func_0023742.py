def get_tag(self, name):
        """Return the tag as Tag object."""
        res = self.get_request('/tag/' + name)
        return Tag(cloud_manager=self, **res['tag'])