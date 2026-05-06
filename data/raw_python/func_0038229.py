def ordered_tags(self):
        """gets the related tags

        :return: `list` of `Tag` instances
        """
        tags = list(self.tags.all())
        return sorted(
            tags,
            key=lambda tag: ((type(tag) != Tag) * 100000) + tag.count(),
            reverse=True
        )