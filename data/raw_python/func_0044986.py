def search_on(self, *fields, **query):
        """
        Search for query on given fields.

        Query modifier can be one of these:
            * exact
            * contains
            * startswith
            * endswith
            * range
            * lte
            * gte

        Args:
            \*fields (str): Field list to be searched on
            \*\*query:  Search query. While it's implemented as \*\*kwargs
             we only support one (first) keyword argument.

        Returns:
            Self. Queryset object.

        Examples:
            >>> Person.objects.search_on('name', 'surname', contains='john')
            >>> Person.objects.search_on('name', 'surname', startswith='jo')
        """
        clone = copy.deepcopy(self)
        clone.adapter.search_on(*fields, **query)
        return clone