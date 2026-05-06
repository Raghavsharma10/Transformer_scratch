def get_alternatives(self):
        """ Get the spelling alternatives for search terms.

            Returns:
                A dict in form:
                {<search term>: {'count': <number of times the searh term occurs in the Storage>,
                                 'words': {<an alternative>: {'count': <number of times the alternative occurs in the Storage>,
                                                              'cr': <cr value of the alternative>,
                                                              'idif': <idif value of the alternative>,
                                                              'h': <h value of the alternative>}
                                          } // Repeated for every alternative.
                                }
                } // Repeated for every search term
        """
        return dict([(alternatives.find('to').text,
                      {'count': int(alternatives.find('count').text),
                       'words': dict([(word.text, word.attrib)
                                      for word in alternatives.findall('word')])})
                     for alternatives in
                     self._content.find('alternatives_list').findall('alternatives')])