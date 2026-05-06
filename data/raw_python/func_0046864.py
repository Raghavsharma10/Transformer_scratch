def add_child_objective_bank(self, objective_bank_id=None, parent_id=None, child_id=None):
        """Adds a child to an objective bank.

        arg:    objective_bank_id (osid.id.Id): the ``Id`` of an
                objective bank
        arg:    child_id (osid.id.Id): the ``Id`` of the new child
        raise:  AlreadyExists - ``objective_bank_id`` is already a
                parent of ``child_id``
        raise:  NotFound - ``objective_bank_id`` or ``child_id`` not
                found
        raise:  NullArgument - ``objective_bank_id`` or ``child_id`` is
                ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        url_path = self._urls.children(alias=objective_bank_id, bank_id=parent_id)
        current_children_ids = self._get_request(url_path)['ids']
        current_children_ids.append(str(child_id))
        new_children_ids = {
            'ids': current_children_ids
        }
        return self._put_request(url_path, new_children_ids)