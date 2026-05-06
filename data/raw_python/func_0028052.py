def _get_stringlist_from_commastring(self, field):
        # type: (str) -> List[str]
        """Return list of strings from comma separated list

        Args:
            field (str): Field containing comma separated list

        Returns:
            List[str]: List of strings
        """
        strings = self.data.get(field)
        if strings:
            return strings.split(',')
        else:
            return list()