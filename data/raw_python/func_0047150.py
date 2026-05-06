def get_root_objectives(self):
        """Gets the root objective in this objective hierarchy.

        return: (osid.learning.ObjectiveList) - the root objective
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method is must be implemented.*

        """
        # Implemented from template for
        # osid.ontology.SubjectHierarchySession.get_root_subjects_template
        root_ids = self._hierarchy_session.get_roots()
        collection = JSONClientValidated('learning',
                                         collection='Objective',
                                         runtime=self._runtime)
        result = collection.find(
            dict({'_id': {'$in': [ObjectId(root_id.get_identifier()) for root_id in root_ids]}},
                 **self._view_filter()))
        return objects.ObjectiveList(
            result,
            runtime=self._runtime,
            proxy=self._proxy)