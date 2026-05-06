def new_table(self, name, add_id=True, **kwargs):
        """
        Create a new table, if it does not exist, or update an existing table if it does
        :param name:  Table name
        :param add_id: If True, add an id field ( default is True )
        :param kwargs: Other options passed to table object
        :return:
        """

        return self.dataset.new_table(name=name, add_id=add_id, **kwargs)