def find(self, selector: str = '*', containing: _Containing = None, first: bool = False, _encoding: str = None) -> _Find:
            """Given a simple element name, returns a list of
            :class:`Element <Element>` objects or a single one.

            :param selector: Element name to find.
            :param containing: If specified, only return elements that contain the provided text.
            :param first: Whether or not to return just the first result.
            :param _encoding: The encoding format.

            If ``first`` is ``True``, only returns the first
            :class:`Element <Element>` found.
            """

            # Convert a single containing into a list.
            if isinstance(containing, str):
                containing = [containing]

            encoding = _encoding or self.encoding
            elements = [
                Element(element=found, default_encoding=encoding)
                for found in self.pq(selector)
            ]

            if containing:
                elements_copy = elements.copy()
                elements = []

                for element in elements_copy:
                    if any([c.lower() in element.text.lower() for c in containing]):
                        elements.append(element)

                elements.reverse()

            return _get_first_or_list(elements, first)