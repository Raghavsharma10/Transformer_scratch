def delete_grade_entry(self, grade_entry_id):
        """Deletes the ``GradeEntry`` identified by the given ``Id``.

        arg:    grade_entry_id (osid.id.Id): the ``Id`` of the
                ``GradeEntry`` to delete
        raise:  NotFound - a ``GradeEntry`` was not found identified by
                the given ``Id``
        raise:  NullArgument - ``grade_entry_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceAdminSession.delete_resource_template
        collection = JSONClientValidated('grading',
                                         collection='GradeEntry',
                                         runtime=self._runtime)
        if not isinstance(grade_entry_id, ABCId):
            raise errors.InvalidArgument('the argument is not a valid OSID Id')
        grade_entry_map = collection.find_one(
            dict({'_id': ObjectId(grade_entry_id.get_identifier())},
                 **self._view_filter()))

        objects.GradeEntry(osid_object_map=grade_entry_map, runtime=self._runtime, proxy=self._proxy)._delete()
        collection.delete_one({'_id': ObjectId(grade_entry_id.get_identifier())})