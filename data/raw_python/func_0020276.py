def words_from_text(self, text, for_search=False):
        '''Generator of indexable words in *text*.
This functions loop through the :attr:`word_middleware` attribute
to process the text.

:param text: string from which to extract words.
:param for_search: flag indicating if the the words will be used for search
    or to index the database. This flug is used in conjunction with the
    middleware flag *for_search*. If this flag is ``True`` (i.e. we need to
    search the database for the words in *text*), only the
    middleware functions in :attr:`word_middleware` enabled for searching are
    used.

    Default: ``False``.

return a *list* of cleaned words.
'''
        if not text:
            return []
        word_gen = self.split_text(text)
        for middleware, fors in self.word_middleware:
            if for_search and not fors:
                continue
            word_gen = middleware(word_gen)
        if isgenerator(word_gen):
            word_gen = list(word_gen)
        return word_gen