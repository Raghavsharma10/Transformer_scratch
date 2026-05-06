def get_by_value(cls, value, type):
        """ Converts a value into a corresponding  data object.
        For files, this looks up a file DataObject by name, uuid, and/or md5.
        For other types, it creates a new DataObject.
        """
        if type == 'file':
            return cls._get_file_by_value(value)
        else:
            data_object = DataObject(data={
                'value': cls._type_cast(value, type)}, type=type)
            data_object.full_clean()
            data_object.save()
            return data_object