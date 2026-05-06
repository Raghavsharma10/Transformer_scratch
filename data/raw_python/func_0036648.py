def add_tags(self, tags):
        """
            Add tags to the comments
        """
        if not isinstance(tags, list):
            tags = [tags]
        self._bugsy.request('bug/comment/%s/tags' % self._comment['id'],
                            method='PUT', json={"add": tags})