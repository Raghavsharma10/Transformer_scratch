def get_tags(self):
        """List all tags as Tag objects."""
        res = self.get_request('/tag')
        return [Tag(cloud_manager=self, **tag) for tag in res['tags']['tag']]