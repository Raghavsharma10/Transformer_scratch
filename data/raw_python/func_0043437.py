def export_deleted_fields(self):
        """
        Resturns a list with any deleted fields form original data.
        In tree models, deleted fields on children will be appended.
        """
        result = self.__deleted_fields__.copy()

        for key, value in self.__original_data__.items():
            if key in result:
                continue
            try:
                partial = value.export_deleted_fields()
                result.extend(['.'.join([key, key2]) for key2 in partial])
            except AttributeError:
                pass

        return result