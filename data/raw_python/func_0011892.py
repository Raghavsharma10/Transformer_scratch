def get_top_display(self, **kwargs):
        '''
        Returns all concepts or collections that form the top-level of a display
        hierarchy.

        As opposed to the :meth:`get_top_concepts`, this method can possibly
        return both concepts and collections.

        :rtype: Returns a list of concepts and collections. For each an
            id is present and a label. The label is determined by looking at
            the `**kwargs` parameter, the default language of the provider
            and falls back to `en` if nothing is present.
        '''
        language = self._get_language(**kwargs)
        url = self.url + '/lijst.json'
        args = {'type[]': ['HR']}
        r = self.session.get(url, params=args)
        result = r.json()
        items = result
        top = self.get_by_id(items[0]['id'])
        res = []
        def expand_coll(res, coll):
            for nid in coll.members:
                c = self.get_by_id(nid)
                res.append({
                    'id': c.id,
                    'label': c.label(language)
                })
            return res
        return expand_coll(res, top)