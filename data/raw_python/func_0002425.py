def _get_list_select(self, column, key=None):
        """
        Get the columns that should be used in a list

        :param column: The column to get the values for
        :type column: str

        :param key: The key
        :type key: str

        :return: The list of values
        :rtype: list
        """
        if key is None:
            elements = [column]
        else:
            elements = [column, key]

        select = []
        for elem in elements:
            dot = elem.find('.')

            if dot >= 0:
                select.append(column[dot + 1:])
            else:
                select.append(elem)

        return select