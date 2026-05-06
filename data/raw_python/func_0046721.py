def get_grade_entries_for_gradebook_column_and_resource(self, gradebook_column_id, resource_id):
        """Gets a ``GradeEntryList`` for the gradebook column and key resource.

        arg:    gradebook_column_id (osid.id.Id): a gradebook column
                ``Id``
        arg:    resource_id (osid.id.Id): a key resource ``Id``
        return: (osid.grading.GradeEntryList) - the returned
                ``GradeEntry`` list
        raise:  NullArgument - ``gradebook_column_id`` or
                ``resource_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.relationship.RelationshipLookupSession.get_relationships_for_peers
        # NOTE: This implementation currently ignores plenary and effective views
        collection = JSONClientValidated('grading',
                                         collection='GradeEntry',
                                         runtime=self._runtime)
        result = collection.find(
            dict({'gradebookColumnId': str(gradebook_column_id),
                  'keyResourceId': str(resource_id)},
                 **self._view_filter())).sort('_id', ASCENDING)
        return objects.GradeEntryList(result, runtime=self._runtime)