def get_nodes(self, id_, ancestor_levels=10, descendant_levels=10, include_siblings=False):
        """Gets a portion of the hierarchy for the given ``Id``.

        arg:    id (osid.id.Id): the ``Id`` to query
        arg:    ancestor_levels (cardinal): the maximum number of
                ancestor levels to include. A value of 0 returns no
                parents in the node.
        arg:    descendant_levels (cardinal): the maximum number of
                descendant levels to include. A value of 0 returns no
                children in the node.
        arg:    include_siblings (boolean): ``true`` to include the
                siblings of the given node, ``false`` to omit the
                siblings
        return: (osid.hierarchy.Node) - a node
        raise:  NotFound - ``id`` is not found
        raise:  NullArgument - ``id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # This impl ignores include_siblings, assumes false
        include_siblings = bool(include_siblings)
        parent_node_list = []
        child_node_list = []
        if ancestor_levels != 0:
            for parent_id in self.get_parents(id_):
                parent_node_list.append(self.get_nodes(parent_id, ancestor_levels - 1, 0))
        if descendant_levels != 0:
            for child_id in self.get_children(id_):
                child_node_list.append(self.get_nodes(child_id, 0, descendant_levels - 1))
        return objects.Node({'type': 'OsidNode',
                             'id': str(id_),
                             'childNodes': child_node_list,
                             'parentNodes': parent_node_list,
                             'root': not self.has_parents(id_),
                             'leaf': not self.has_children(id_),
                             'sequestered': False})