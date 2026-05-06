def update_relationship(self, relationship_form):
        """Updates an existing relationship.

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
        # Implemented from template for
        # osid.resource.ResourceAdminSession.update_resource_template
        collection = JSONClientValidated('relationship',
                                         collection='Relationship',
                                         runtime=self._runtime)
        if not isinstance(relationship_form, ABCRelationshipForm):
            raise errors.InvalidArgument('argument type is not an RelationshipForm')
        if not relationship_form.is_for_update():
            raise errors.InvalidArgument('the RelationshipForm is for update only, not create')
        try:
            if self._forms[relationship_form.get_id().get_identifier()] == UPDATED:
                raise errors.IllegalState('relationship_form already used in an update transaction')
        except KeyError:
            raise errors.Unsupported('relationship_form did not originate from this session')
        if not relationship_form.is_valid():
            raise errors.InvalidArgument('one or more of the form elements is invalid')
        collection.save(relationship_form._my_map)

        self._forms[relationship_form.get_id().get_identifier()] = UPDATED

        # Note: this is out of spec. The OSIDs don't require an object to be returned:
        return objects.Relationship(
            osid_object_map=relationship_form._my_map,
            runtime=self._runtime,
            proxy=self._proxy)