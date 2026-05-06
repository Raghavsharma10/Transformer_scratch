def startswith(self, name: str) -> List[str]:
        """Return a list of all keywords starting with the given string.

        >>> from hydpy.core.devicetools import Keywords
        >>> keywords = Keywords('first_keyword', 'second_keyword',
        ...                     'keyword_3', 'keyword_4',
        ...                     'keyboard')
        >>> keywords.startswith('keyword')
        ['keyword_3', 'keyword_4']
        """
        return sorted(keyword for keyword in self if keyword.startswith(name))