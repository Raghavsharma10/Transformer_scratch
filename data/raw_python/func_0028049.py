def _get_tags(self):
        # type: () -> List[str]
        """Return the dataset's list of tags

        Returns:
            List[str]: list of tags or [] if there are none
        """
        tags = self.data.get('tags', None)
        if not tags:
            return list()
        return [x['name'] for x in tags]