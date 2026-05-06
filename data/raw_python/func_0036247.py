def from_keywords(self, keyword_list, strictness=2, timeout=3):
        """
        Generate a sentence from initial_list.

        :param list keyword_list: a list of keywords to be included in the sentence.
        :param int | None strictness: None for highest strictness. 2 or 1 for a less strict POS matching
        :param float timeout: timeout of this function
        :return list of tuple:

        >>> ToSentence().from_keywords(['gains', 'grew', 'pass', 'greene', 'escort', 'illinois'])
        [('The', False), ('gains', True), ('of', False), ('Bienville', False), ('upon', False), ('grew', True), ('liberal', False), ('pass', True), ('to', False), ('the', False), ('Indians', False), (',', False), ('in', False), ('greene', True), ('to', False), ('drive', False), ('back', False), ('the', False), ('Carolina', False), ('escort', True), (',', False), ('was', False), ('probably', False), ('a', False), ('illinois', True)]
        """
        return self.keyword_parse.from_keyword_list(keyword_list, strictness, timeout)