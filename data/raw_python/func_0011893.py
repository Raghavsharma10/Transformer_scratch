def get_children_display(self, id, **kwargs):
        '''
        Return a list of concepts or collections that should be displayed
        under this concept or collection.

        :param id: A concept or collection id.
        :rtype: A list of concepts and collections. For each an
            id is present and a label. The label is determined by looking at
            the `**kwargs` parameter, the default language of the provider
            and falls back to `en` if nothing is present. If the id does not
            exist, return `False`.
        '''
        language = self._get_language(**kwargs)
        item = self.get_by_id(id)
        res = []
        if isinstance(item, Collection):
            for mid in item.members:
                m = self.get_by_id(mid)
                res.append({
                    'id': m.id,
                    'label': m.label(language)
                })
        else:
            for cid in item.narrower:
                c = self.get_by_id(cid)
                res.append({
                    'id': c.id,
                    'label': c.label(language)
                })
        return res