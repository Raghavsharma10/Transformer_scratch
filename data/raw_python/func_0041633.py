def next_listing(self, limit=None):
        """GETs next :class:`Listing` directed to by this :class:`Listing`.  Returns :class:`Listing` object.
        
        :param limit: max number of entries to get
        :raise UnsupportedError: raised when trying to load more comments
        """
        if self.after:
            return self._reddit._limit_get(self._path, params={'after': self.after}, limit=limit or self._limit)
        elif self._has_literally_more:
            more = self[-1]
            data = dict(
                link_id=self[0].parent_id,
                id=more.name,
                children=','.join(more.children)
            )
            j = self._reddit.post('api', 'morechildren', data=data)
            # since reddit is inconsistent here, we're hacking it to be
            # consistent so it'll work with _thingify
            d = j['json']
            d['kind'] = 'Listing'
            d['data']['children'] = d['data']['things']
            del d['data']['things']
            return self._reddit._thingify(d, path=self._path) 
        else:
            raise NoMoreError('no more items')