def _split_list(cls, items, separator=",", last_separator=" and "):
        """
        Splits a string listing elements into an actual list.

        Parameters
        ----------
        items: :class:`str`
            A string listing elements.
        separator: :class:`str`
            The separator between each item. A comma by default.
        last_separator: :class:`str`
            The separator used for the last item. ' and ' by default.

        Returns
        -------
        :class:`list` of :class:`str`
            A list containing each one of the items.
        """
        if items is None:
            return None
        items = items.split(separator)
        last_item = items[-1]
        last_split = last_item.split(last_separator)
        if len(last_split) > 1:
            items[-1] = last_split[0]
            items.append(last_split[1])
        return [e.strip() for e in items]