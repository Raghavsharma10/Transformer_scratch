def delete_gradebook_column(self, gradebook_column_id):
        """Deletes the ``GradebookColumn`` identified by the given ``Id``.

        arg:    gradebook_column_id (osid.id.Id): the ``Id`` of the
                ``GradebookColumn`` to delete
        raise:  NotFound - a ``GradebookColumn`` was not found
                identified by the given ``Id``
        raise:  NullArgument - ``gradebook_column_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        if not isinstance(gradebook_column_id, ABCId):
            raise errors.InvalidArgument('the argument is not a valid OSID Id')

        # check that no entries already exist for this gradebook column
        grading_manager = self._get_provider_manager('GRADING')
        gels = grading_manager.get_grade_entry_lookup_session(proxy=getattr(self, "_proxy", None))
        gels.use_federated_gradebook_view()
        entries = gels.get_grade_entries_for_gradebook_column(gradebook_column_id)
        if self._has_entries(gradebook_column_id):
            raise errors.IllegalState('Entries exist in this gradebook column. Cannot delete it.')

        collection = JSONClientValidated('grading',
                                         collection='GradebookColumn',
                                         runtime=self._runtime)

        gradebook_column_map = collection.find_one({'_id': ObjectId(gradebook_column_id.get_identifier())})

        objects.GradebookColumn(osid_object_map=gradebook_column_map,
                                runtime=self._runtime,
                                proxy=self._proxy)._delete()
        collection.delete_one({'_id': ObjectId(gradebook_column_id.get_identifier())})