def alias_sequence_rule(self, sequence_rule_id, alias_id):
        """Adds a ``Id`` to a ``SequenceRule`` for the purpose of creating compatibility.

        The primary ``Id`` of the ``SequenceRule`` is determined by the
        provider. The new ``Id`` performs as an alias to the primary
        ``Id`` . If the alias is a pointer to another sequence rule. it
        is reassigned to the given sequence rule ``Id``.

        arg:    sequence_rule_id (osid.id.Id): the ``Id`` of a
                ``SequenceRule``
        arg:    alias_id (osid.id.Id): the alias ``Id``
        raise:  AlreadyExists - ``alias_id`` is already assigned
        raise:  NotFound - ``sequence_rule_id`` not found
        raise:  NullArgument - ``sequence_rule_id`` or ``alias_id`` is
                ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceAdminSession.alias_resources_template
        self._alias_id(primary_id=sequence_rule_id, equivalent_id=alias_id)