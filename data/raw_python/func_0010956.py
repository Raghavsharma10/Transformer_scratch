def tags(self):
        """ Returns a dict containing tags and their localized labels as values """
        return dict([(t, self._catalog.tags.get(t, t)) for t in self._asset.get("tags", [])])