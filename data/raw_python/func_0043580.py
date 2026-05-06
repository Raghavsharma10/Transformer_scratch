def export_deleted_fields(self):
        """
        Returns a list with any deleted fields form original data.
        In tree models, deleted fields on children will be appended.
        """
        result = []
        if self.__modified_data__ is not None:
            return result

        for index, item in enumerate(self):
            try:
                deleted_fields = item.export_deleted_fields()
                result.extend(['{}.{}'.format(index, key) for key in deleted_fields])
            except AttributeError:
                pass
        return result