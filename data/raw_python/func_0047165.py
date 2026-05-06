def get_dependent_objectives(self, objective_id):
        """Gets a list of ``Objectives`` that require the given ``Objective``.

        In plenary mode, the returned list contains all of the immediate
        requisites, or an error results if an Objective is not found or
        inaccessible. Otherwise, inaccessible ``Objectives`` may be
        omitted from the list and may present the elements in any order
        including returning a unique set.

        arg:    objective_id (osid.id.Id): ``Id`` of the ``Objective``
        return: (osid.learning.ObjectiveList) - the returned
                ``Objective`` list
        raise:  NotFound - ``objective_id`` not found
        raise:  NullArgument - ``objective_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.learning.ObjectiveRequisiteSession.get_dependent_objectives_template
        # NOTE: This implementation currently ignores plenary view
        requisite_type = Type(**Relationship().get_type_data('OBJECTIVE.REQUISITE'))
        relm = self._get_provider_manager('RELATIONSHIP')
        rls = relm.get_relationship_lookup_session(proxy=self._proxy)
        rls.use_federated_family_view()
        requisite_relationships = rls.get_relationships_by_genus_type_for_destination(objective_id,
                                                                                      requisite_type)
        source_ids = [ObjectId(r.get_source_id().identifier)
                      for r in requisite_relationships]
        collection = JSONClientValidated('learning',
                                         collection='Objective',
                                         runtime=self._runtime)
        result = collection.find({'_id': {'$in': source_ids}})
        return objects.ObjectiveList(result, runtime=self._runtime)