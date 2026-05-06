def delete_attr_by_path(self, field):
        """
        Function for deleting a field specifying the path in the whole model as described
        in :func:`dirty:models.models.BaseModel.perform_function_by_path`
        """
        index_list, next_field = self._get_indexes_by_path(field)
        if index_list:
            for index in reversed(index_list):
                if next_field:
                    self[index].delete_attr_by_path(next_field)
                else:
                    self.pop(index)