def search(self, name, value):
        """
        Searches the table for the entry specified by name
        and value

        Returns one of the following:
            - ``None``, no match at all
            - ``(index, name, None)`` for partial matches on name only.
            - ``(index, name, value)`` for perfect matches.
        """
        partial = None

        header_name_search_result = CocaineHeaders.STATIC_TABLE_MAPPING.get(name)
        if header_name_search_result:
            index = header_name_search_result[1].get(value)
            if index is not None:
                return index, name, value
            partial = (header_name_search_result[0], name, None)

        offset = len(CocaineHeaders.STATIC_TABLE)
        for (i, (n, v)) in enumerate(self.dynamic_entries):
            if n == name:
                if v == value:
                    return i + offset + 1, n, v
                elif partial is None:
                    partial = (i + offset + 1, n, None)
        return partial