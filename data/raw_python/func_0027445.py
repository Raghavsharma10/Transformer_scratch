def up(self, count=1, to_name=None):
        """
        :return: a builder representing an ancestor of the current element,
                 by default the parent element.

        :param count: return the n'th ancestor element; defaults to 1 which
            means the immediate parent. If *count* is greater than the number
            of number of ancestors return the document's root element.
        :type count: integer >= 1 or None
        :param to_name: return the nearest ancestor element with the matching
            name, or the document's root element if there are no matching
            elements. This argument trumps the ``count`` argument.
        :type to_name: string or None
        """
        elem = self._element
        up_count = 0
        while True:
            # Don't go up beyond the document root
            if elem.is_root or elem.parent is None:
                break
            elem = elem.parent
            if to_name is None:
                up_count += 1
                if up_count >= count:
                    break
            else:
                if elem.name == to_name:
                    break
        return Builder(elem)