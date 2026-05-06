def find_or_new(self, id, columns=None):
        """
        Find a model by its primary key or return new instance of the related model.

        :param id: The primary key
        :type id: mixed

        :param columns:  The columns to retrieve
        :type columns: list

        :rtype: Collection or Model
        """
        if columns is None:
            columns = ['*']

        instance = self.find(id, columns)

        if instance is None:
            instance = self._related.new_instance()
            self._set_foreign_attributes_for_create(instance)

        return instance