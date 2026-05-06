def search_bugs(self, terms):
        '''http://bugzilla.readthedocs.org/en/latest/api/core/v1/bug.html#search-bugs
        terms = [{'product': 'Infrastructure & Operations'}, {'status': 'NEW'}]'''
        params = ''
        for i in terms:
            k = i.popitem()
            params = '{p}&{new}={value}'.format(p=params, new=quote_url(k[0]),
                        value=quote_url(k[1]))
        return DotDict(self._get('bug', params=params))