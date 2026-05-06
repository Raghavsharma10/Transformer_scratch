def get_types(self):
        """Gets all the known Types.

        return: (osid.type.TypeList) - the list of all known Types
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        compliance: mandatory - This method must be implemented.

        """
        from . import types
        url_path = '/handcar/services/learning/types/'
        type_list = self._get_request(url_path)
        type_list += types.TYPES
        return objects.TypeList(type_list)