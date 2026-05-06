def reset_attr_by_path(self, field):
        """
        Function for restoring a field specifying the path in the whole model as described
        in :func:`dirty:models.models.BaseModel.perform_function_by_path`
        """
        index_list, next_field = self._get_indexes_by_path(field)
        if index_list:
            if next_field:
                for index in index_list:
                    self[index].reset_attr_by_path(next_field)
            else:
                for index in index_list:
                    try:
                        self[index].clear_modified_data()
                    except (AttributeError, IndexError):
                        return