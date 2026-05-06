def search(self, template: str, first: bool = False) -> _Result:
        """Search the :class:`Element <Element>` for the given parse
        template.

        :param template: The Parse template to use.
        """
        elements = [r for r in findall(template, self.xml)]

        return _get_first_or_list(elements, first)