def get_objectives_by_ids(self, objective_ids=None):
        """Gets an ObjectiveList corresponding to the given IdList.
        In plenary mode, the returned list contains all of the
        objectives specified in the Id list, in the order of the list,
        including duplicates, or an error results if an Id in the
        supplied list is not found or inaccessible. Otherwise,
        inaccessible Objectives may be omitted from the list and may
        present the elements in any order including returning a unique
        set.
        arg:    objectiveIds (osid.id.IdList): the list of Ids to
                retrieve
        return: (osid.learning.ObjectiveList) - the returned Objective
                list
        raise:  NotFound - an Id was not found
        raise:  NullArgument - objectiveIds is null
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        compliance: mandatory - This method must be implemented.

        """
        if objective_ids is None:
            raise NullArgument()

        url_path = construct_url('objectives_by_ids',
                                 obj_ids=objective_ids)
        objectives = self._get_request(url_path)

        # for i in objective_ids:
        #     objective = None
        #     url_path = construct_url('objectives',
        #                              obj_id=i)
        #     try:
        #         objective = self._get_request(url_path)
        #     except (NotFound, OperationFailed):
        #         if self._objective_view == PLENARY:
        #             raise
        #         else:
        #             pass
        #     if objective:
        #         if not (self._objective_view == COMPARATIVE and
        #                 objective in objectives):
        #             objectives.append(objective)
        return objects.ObjectiveList(objectives)