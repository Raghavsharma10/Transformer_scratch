def get_relationship_form_for_create(self, source_id, destination_id, relationship_record_types):
        """Gets the relationship form for creating new relationships.

        A new form should be requested for each create transaction.

        arg:    source_id (osid.id.Id): ``Id`` of a peer
        arg:    destination_id (osid.id.Id): ``Id`` of the related peer
        arg:    relationship_record_types (osid.type.Type[]): array of
                relationship record types
        return: (osid.relationship.RelationshipForm) - the relationship
                form
        raise:  NotFound - ``source_id`` or ``destination_id`` is not
                found
        raise:  NullArgument - ``source_id`` or ``destination_id`` is
                ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - unable to get form for requested recod
                types
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.relationship.RelationshipAdminSession.get_relationship_form_for_create
        # These really need to be in module imports:
        from dlkit.abstract_osid.id.primitives import Id as ABCId
        from dlkit.abstract_osid.type.primitives import Type as ABCType
        if not isinstance(source_id, ABCId):
            raise errors.InvalidArgument('argument is not a valid OSID Id')
        if not isinstance(destination_id, ABCId):
            raise errors.InvalidArgument('argument is not a valid OSID Id')
        for arg in relationship_record_types:
            if not isinstance(arg, ABCType):
                raise errors.InvalidArgument('one or more argument array elements is not a valid OSID Type')
        if relationship_record_types == []:
            # WHY are we passing family_id = self._catalog_id below, seems redundant:
            obj_form = objects.RelationshipForm(
                family_id=self._catalog_id,
                source_id=source_id,
                destination_id=destination_id,
                catalog_id=self._catalog_id,
                runtime=self._runtime,
                proxy=self._proxy)
        else:
            obj_form = objects.RelationshipForm(
                family_id=self._catalog_id,
                record_types=relationship_record_types,
                source_id=source_id,
                destination_id=destination_id,
                catalog_id=self._catalog_id,
                runtime=self._runtime,
                proxy=self._proxy)
        obj_form._for_update = False
        self._forms[obj_form.get_id().get_identifier()] = not CREATED
        return obj_form