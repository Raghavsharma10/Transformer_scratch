def add_child(self, id_, child_id):
        """Adds a child to a ``Id``.

        arg:    id (osid.id.Id): the ``Id`` of the node
        arg:    child_id (osid.id.Id): the ``Id`` of the new child
        raise:  AlreadyExists - ``child_id`` is already a child of
                ``id``
        raise:  NotFound - ``id`` or ``child_id`` not found
        raise:  NullArgument - ``id`` or ``child_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        if bool(self._rls.get_relationships_by_genus_type_for_peers(id_, child_id, self._relationship_type).available()):
            raise errors.AlreadyExists()
        rfc = self._ras.get_relationship_form_for_create(id_, child_id, [])
        rfc.set_display_name(str(id_) + ' to ' + str(child_id) + ' Parent-Child Relationship')
        rfc.set_description(self._relationship_type.get_display_name().get_text() + ' relationship for parent: ' + str(id_) + ' and child: ' + str(child_id))
        rfc.set_genus_type(self._relationship_type)
        self._ras.create_relationship(rfc)