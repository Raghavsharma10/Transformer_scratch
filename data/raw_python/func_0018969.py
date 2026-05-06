def endswith(self, name: str) -> List[str]:
        """Return a list of all keywords ending with the given string.

        >>> from hydpy.core.devicetools import Keywords
        >>> keywords = Keywords('first_keyword', 'second_keyword',
        ...                     'keyword_3', 'keyword_4',
        ...                     'keyboard')
        >>> keywords.endswith('keyword')
        ['first_keyword', 'second_keyword']
        """
        return sorted(keyword for keyword in self if keyword.endswith(name))