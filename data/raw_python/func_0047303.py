def get_type(self, namespace=None, identifier=None, authority=None):
        """Gets a Type by its string representation which is a combination
        of the authority and identifier.

        This method only returns the Type if it is known by the given
        identification components.

        arg:    namespace (string): the identifier namespace
        arg:    identifier (string): the identifier
        arg:    authority (string): the authority
        return: (osid.type.Type) - the Type
        raise:  NotFound - the type is not found
        raise:  NullArgument - null argument provided
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        compliance: mandatory - This method must be implemented.

        """
        from . import types
        from ..osid.osid_errors import NotFound, NullArgument
        if namespace is None or identifier is None or authority is None:
            raise NullArgument()
        type_identifier = namespace + '%3A' + identifier + '%40' + authority
        url_path = '/handcar/services/learning/types/' + type_identifier
        try:
            result = self._get_request(url_path)
        except NotFound:
            result = None
            for t in types.TYPES:
                if t['id'] == type_identifier:
                    result = t
            if result is None:
                raise NotFound()
        return Type(result)