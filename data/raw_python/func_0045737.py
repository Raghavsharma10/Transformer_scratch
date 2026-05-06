def create_relationship(self, relationship_form=None):
        """Creates a new ``Relationship``.

        arg:    relationship_form (osid.relationship.RelationshipForm):
                the form for this ``Relationship``
        return: (osid.relationship.Relationship) - the new
                ``Relationship``
        raise:  IllegalState - ``relationship_form`` already used in a
                create transaction
        raise:  InvalidArgument - one or more of the form elements is
                invalid
        raise:  NullArgument - ``relationship_form`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - ``relationship_form`` did not originate
                from ``get_relationship_form_for_create()``
        *compliance: mandatory -- This method must be implemented.*

        """
        if relationship_form is None:
            raise NullArgument()
        if not isinstance(relationship_form, abc_relationship_objects.RelationshipForm):
            raise InvalidArgument('argument type is not a RelationshipForm')
        if relationship_form.is_for_update():
            raise InvalidArgument('form is for update only, not create')
        try:
            if self._forms[relationship_form.get_id().get_identifier()] == CREATED:
                raise IllegalState('form already used in a create transaction')
        except KeyError:
            raise Unsupported('form did not originate from this session')
        if not relationship_form.is_valid():
            raise InvalidArgument('one or more of the form elements is invalid')

        url_path = ('/handcar/services/relationship/families/' +
                    self._catalog_idstr + '/relationships')
        try:
            result = self._post_request(url_path, relationship_form._my_map)
        except Exception:
            raise  # OperationFailed
        self._forms[relationship_form.get_id().get_identifier()] = CREATED
        return objects.Relationship(result)