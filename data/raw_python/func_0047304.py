def has_type(self, type_=None):
        """Tests if the given Type is known.

        arg:    type (osid.type.Type): the Type to look for
        return: (boolean) - true if the given Type is known, false
                otherwise
        raise:  NullArgument - type is null
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        compliance: mandatory - This method must be implemented.

        """
        # This seems silly :)
        return bool(self.get_type(namespace=type_.get_namespace(),
                                  identifier=type_.get_identifier(),
                                  authority=type_.get_authority()))