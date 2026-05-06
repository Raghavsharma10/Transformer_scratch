def search(self, text, lookup=None):
        '''Search *text* in model. A search engine needs to be installed
for this function to be available.

:parameter text: a string to search.
:return type: a new :class:`Query` instance.
'''
        q = self._clone()
        q.text = (text, lookup)
        return q