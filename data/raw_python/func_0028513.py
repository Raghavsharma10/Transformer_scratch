def to_d(self):
        """
        Args: self

        Returns: a d that stems from self

        Raises:
            ValueError: dictionary update sequence element #index has length
            len(tuple); 2 is required

            TypeError: cannot convert dictionary update sequence element
            #index to a sequence
        """

        try:
            return ww.d(self)
        except (TypeError, ValueError):

            for i, element in enumerate(self):
                try:
                    iter(element)

                # TODO: find out why we can't cover this branch. The code
                # is tested but don't appear in coverage
                except TypeError:  # pragma: no cover
                    # TODO: use raise_from ?
                    raise ValueError(("'{}' (position {}) is not iterable. You"
                                      " can only create a dictionary from a "
                                      "elements that are iterables, such as "
                                      "tuples, lists, etc.")
                                     .format(element, i))

                try:
                    size = len(element)
                except TypeError:  # ignore generators, it's already consummed
                    pass
                else:
                    raise ValueError(("'{}' (position {}) contains {} "
                                      "elements. You can only create a "
                                      "dictionary from iterables containing "
                                      "2 elements.").format(element, i, size))

            raise