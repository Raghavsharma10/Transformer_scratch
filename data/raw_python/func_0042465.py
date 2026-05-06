def search_tags(self, tags):
        """
        Search assets by passing a list of one or more tags.
        """
        qs = self.filter(tags__name__in=tags).order_by('file').distinct()
        return qs