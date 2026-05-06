def contains(self, name: str) -> List[str]:
        """Return a list of all keywords containing the given string.

        >>> from hydpy.core.devicetools import Keywords
        >>> keywords = Keywords('first_keyword', 'second_keyword',
        ...                     'keyword_3', 'keyword_4',
        ...                     'keyboard')
        >>> keywords.contains('keyword')
        ['first_keyword', 'keyword_3', 'keyword_4', 'second_keyword']
        """
        return sorted(keyword for keyword in self if name in keyword)