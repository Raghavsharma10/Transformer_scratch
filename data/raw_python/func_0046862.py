def add_root_objective_bank(self, alias=None, objective_bank_id=None):
        """Adds a root objective bank.

        arg:    objective_bank_id (osid.id.Id): the ``Id`` of an
                objective bank
        raise:  AlreadyExists - ``objective_bank_id`` is already in
                hierarchy
        raise:  NotFound - ``objective_bank_id`` not found
        raise:  NullArgument - ``objective_bank_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        url_path = self._urls.roots(alias=alias)
        current_root_ids = self._get_request(url_path)['ids']
        current_root_ids.append(str(objective_bank_id))
        new_root_ids = {
            'ids': current_root_ids
        }
        return self._put_request(url_path, new_root_ids)