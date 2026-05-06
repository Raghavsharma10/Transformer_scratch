def _extract_relations(self, attribute: str) -> Tuple[List[str], str]:
        """Split and return the list of relation(s) and the attribute.

        :param attribute:
        :type attribute: str

        :return: A tuple where the first element is the list of related
            entities and the second is the attribute.
        :rtype: Tuple[List[str], str]
        """
        splitted = attribute.split(self._attr_sep)
        return (splitted[:-1], splitted[-1])