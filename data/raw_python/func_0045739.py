def update_relationship(self, relationship_id=None, relationship_form=None):
        """Updates an existing relationship.

        arg:    relationship_id (osid.id.Id): the ``Id`` of the
                ``Relationship``
        arg:    relationship_form (osid.relationship.RelationshipForm):
                the form containing the elements to be updated
        raise:  IllegalState - ``relationship_form`` already used in an
                update transaction
        raise:  InvalidArgument - the form contains an invalid value
        raise:  NullArgument - ``relationship_form`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - ``relationship_form`` did not originate
                from ``get_relationship_form_for_update()``
        *compliance: mandatory -- This method must be implemented.*

        """
        if relationship_id is None or relationship_form is None:
            raise NullArgument()
        if not isinstance(relationship_form, objects.RelationshipForm):
            raise InvalidArgument('argument type is not an RelationshipForm')
        if not relationship_form.is_for_update():
            raise InvalidArgument('form is for create only, not update')
        try:
            if self._forms[relationship_form.get_id().get_identifier()] == UPDATED:
                raise IllegalState('form already used in an update transaction')
        except KeyError:
            raise Unsupported('form did not originate from this session')
        if not relationship_form.is_valid():
            raise InvalidArgument('one or more of the form elements is invalid')

        url_path = ('/handcar/services/relationship/families/' +
                    self._catalog_idstr + '/relationships')
        try:
            result = self._put_request(url_path, relationship_form._my_map)
        except Exception:
            raise  # OperationFailed
        self._forms[relationship_form.get_id().get_identifier()] = UPDATED
        return objects.Relationship(result)