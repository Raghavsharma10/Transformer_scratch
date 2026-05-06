def delete_grade_system(self, grade_system_id):
        """Deletes a ``GradeSystem``.

        arg:    grade_system_id (osid.id.Id): the ``Id`` of the
                ``GradeSystem`` to remove
        raise:  NotFound - ``grade_system_id`` not found
        raise:  NullArgument - ``grade_system_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        collection = JSONClientValidated('grading',
                                         collection='GradeSystem',
                                         runtime=self._runtime)
        if not isinstance(grade_system_id, ABCId):
            raise errors.InvalidArgument('the argument is not a valid OSID Id')
        grade_system_map = collection.find_one({'_id': ObjectId(grade_system_id.get_identifier())})

        # check if has columns first
        if self._has_columns(grade_system_id):
            raise errors.InvalidArgument('Grade system being used by gradebook columns. ' +
                                         'Cannot delete it.')

        objects.GradeSystem(osid_object_map=grade_system_map,
                            runtime=self._runtime,
                            proxy=self._proxy)._delete()
        collection.delete_one({'_id': ObjectId(grade_system_id.get_identifier())})