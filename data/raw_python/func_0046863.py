def remove_root_objective_bank(self, alias=None, objective_bank_id=None):
        """Removes a root objective bank.

        arg:    objective_bank_id (osid.id.Id): the ``Id`` of an
                objective bank
        raise:  NotFound - ``objective_bank_id`` is not a root
        raise:  NullArgument - ``objective_bank_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        url_path = self._urls.roots(alias=alias)
        current_root_ids = self._get_request(url_path)['ids']
        modified_list = []
        for root_id in current_root_ids:
            if root_id != str(objective_bank_id):
                modified_list.append(root_id)
        new_root_ids = {
            'ids': modified_list
        }
        return self._put_request(url_path, new_root_ids)