def _get_term_by_id(self, id):
        '''Simple utility function to load a term.
        '''
        url = (self.url + '/%s.json') % id
        r = self.session.get(url)
        return r.json()