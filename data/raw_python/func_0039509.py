def get_words(self):
        """ Get words matching the request search terms.

            Returns:
                A dict in form:
                    {<search term>: {<matching word>: <number of times this word is found in the Storage>
                                    } // Repeated for every matching word.
                    } // Repeated for every search term.
        """
        return dict([(word_list.attrib['to'], dict([(word.text, word.attrib['count'])
                                                    for word in word_list.findall('word')]))
                     for word_list in self._content.findall('list')])