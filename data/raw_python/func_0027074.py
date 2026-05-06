def drop_columns(
        self, max_na_values: int = None, max_unique_values: int = None
    ):
        """
        When max_na_values was informed, remove columns when the proportion of
        total NA values more than max_na_values threshold.

        When max_unique_values was informed, remove columns when the proportion
        of the total of unique values is more than the max_unique_values
        threshold, just for columns with type as object or category.

        :param max_na_values: proportion threshold of max na values
        :param max_unique_values:
        :return:
        """
        step = {}

        if max_na_values is not None:
            step = {
                'data-set': self.iid,
                'operation': 'drop-na',
                'expression': '{"max_na_values":%s, "axis": 1}' % max_na_values
            }
        if max_unique_values is not None:
            step = {
                'data-set': self.iid,
                'operation': 'drop-unique',
                'expression': '{"max_unique_values":%s}' % max_unique_values
            }
        self.attr_update(attr='steps', value=[step])