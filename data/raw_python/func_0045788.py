def reassign_log_entry_to_log(self, log_entry_id, from_log_id, to_log_id):
        """Moves a ``LogEntry`` from one ``Log`` to another.

        Mappings to other ``Logs`` are unaffected.

        arg:    log_entry_id (osid.id.Id): the ``Id`` of the
                ``LogEntry``
        arg:    from_log_id (osid.id.Id): the ``Id`` of the current
                ``Log``
        arg:    to_log_id (osid.id.Id): the ``Id`` of the destination
                ``Log``
        raise:  NotFound - ``log_entry_id, from_log_id,`` or
                ``to_log_id`` not found or ``log_entry_id`` not mapped
                to ``from_log_id``
        raise:  NullArgument - ``log_entry_id, from_log_id,`` or
                ``to_log_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinAssignmentSession.reassign_resource_to_bin
        self.assign_log_entry_to_log(log_entry_id, to_log_id)
        try:
            self.unassign_log_entry_from_log(log_entry_id, from_log_id)
        except:  # something went wrong, roll back assignment to to_log_id
            self.unassign_log_entry_from_log(log_entry_id, to_log_id)
            raise