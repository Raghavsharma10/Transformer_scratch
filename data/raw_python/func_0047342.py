def remove_children(self, id_):
        """Removes all childrenfrom an ``Id``.

        arg:    id (osid.id.Id): the ``Id`` of the node
        raise:  NotFound - an node identified by the given ``Id`` was
                not found
        raise:  NullArgument - ``id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        results = self._rls.get_relationships_by_genus_type_for_source(id_, self._relationship_type)
        if results.available() == 0:
            raise errors.NotFound()
        for r in results:
            self._ras.delete_relationship(r.get_id())