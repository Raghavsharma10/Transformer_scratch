def get_all_requisite_objectives(self, objective_id=None):
        """Gets a list of Objectives that are the requisites for the given
        Objective including the requistes of the requisites, and so on.

        In plenary mode, the returned list contains all of the immediate
        requisites, or an error results if an Objective is not found or
        inaccessible. Otherwise, inaccessible Objectives may be omitted
        from the list and may present the elements in any order
        including returning a unique set.

        arg:    objective_id (osid.id.Id): Id of the Objective
        return: (osid.learning.ObjectiveList) - the returned Objective
                list
        raise:  NotFound - objective_id not found
        raise:  NullArgument - objective_id is null
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        compliance: mandatory - This method must be implemented.

        """
        # This should be re-implemented if and when handcar supports
        # getting all requisites directly
        requisites = list()
        requisite_ids = list()
        all_requisites = self._get_requisites_recursively(objective_id, requisites, requisite_ids)
        return objects.ObjectiveList(all_requisites)