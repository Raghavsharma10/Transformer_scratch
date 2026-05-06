def get_assignable_gradebook_ids(self, gradebook_id):
        """Gets a list of gradebooks including and under the given gradebook node in which any grade system can be assigned.

        arg:    gradebook_id (osid.id.Id): the ``Id`` of the
                ``Gradebook``
        return: (osid.id.IdList) - list of assignable gradebook ``Ids``
        raise:  NullArgument - ``gradebook_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinAssignmentSession.get_assignable_bin_ids
        # This will likely be overridden by an authorization adapter
        mgr = self._get_provider_manager('GRADING', local=True)
        lookup_session = mgr.get_gradebook_lookup_session(proxy=self._proxy)
        gradebooks = lookup_session.get_gradebooks()
        id_list = []
        for gradebook in gradebooks:
            id_list.append(gradebook.get_id())
        return IdList(id_list)