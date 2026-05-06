def add_word_middleware(self, middleware, for_search=True):
        '''Add a *middleware* function to the list of :attr:`word_middleware`,
for preprocessing words to be indexed.

:param middleware: a callable receving an iterable over words.
:param for_search: flag indicating if the *middleware* can be used for the
    text to search. Default: ``True``.
'''
        if hasattr(middleware, '__call__'):
            self.word_middleware.append((middleware, for_search))