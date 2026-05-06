def create_relationship(self, relationship_form):
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
        # Implemented from template for
        # osid.resource.ResourceAdminSession.create_resource_template
        collection = JSONClientValidated('relationship',
                                         collection='Relationship',
                                         runtime=self._runtime)
        if not isinstance(relationship_form, ABCRelationshipForm):
            raise errors.InvalidArgument('argument type is not an RelationshipForm')
        if relationship_form.is_for_update():
            raise errors.InvalidArgument('the RelationshipForm is for update only, not create')
        try:
            if self._forms[relationship_form.get_id().get_identifier()] == CREATED:
                raise errors.IllegalState('relationship_form already used in a create transaction')
        except KeyError:
            raise errors.Unsupported('relationship_form did not originate from this session')
        if not relationship_form.is_valid():
            raise errors.InvalidArgument('one or more of the form elements is invalid')
        insert_result = collection.insert_one(relationship_form._my_map)

        self._forms[relationship_form.get_id().get_identifier()] = CREATED
        result = objects.Relationship(
            osid_object_map=collection.find_one({'_id': insert_result.inserted_id}),
            runtime=self._runtime,
            proxy=self._proxy)

        return result