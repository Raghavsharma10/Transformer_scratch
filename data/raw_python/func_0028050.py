def _add_tag(self, tag):
        # type: (str) -> bool
        """Add a tag

        Args:
            tag (str): Tag to add

        Returns:
            bool: True if tag added or False if tag already present
        """
        tags = self.data.get('tags', None)
        if tags:
            if tag in [x['name'] for x in tags]:
                return False
        else:
            tags = list()
        tags.append({'name': tag})
        self.data['tags'] = tags
        return True