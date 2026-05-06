def delete_grade(self, grade_id):
        """Deletes a ``Grade``.

        arg:    grade_id (osid.id.Id): the ``Id`` of the ``Grade`` to
                remove
        raise:  NotFound - ``grade_id`` not found
        raise:  NullArgument - ``grade_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.repository.AssetAdminSession.delete_asset_content_template
        from dlkit.abstract_osid.id.primitives import Id as ABCId
        from .objects import Grade
        collection = JSONClientValidated('grading',
                                         collection='GradeSystem',
                                         runtime=self._runtime)
        if not isinstance(grade_id, ABCId):
            raise errors.InvalidArgument('the argument is not a valid OSID Id')
        grade_system = collection.find_one({'grades._id': ObjectId(grade_id.get_identifier())})

        index = 0
        found = False
        for i in grade_system['grades']:
            if i['_id'] == ObjectId(grade_id.get_identifier()):
                grade_map = grade_system['grades'].pop(index)
            index += 1
            found = True
        if not found:
            raise errors.OperationFailed()
        Grade(
            osid_object_map=grade_map,
            runtime=self._runtime,
            proxy=self._proxy)._delete()
        collection.save(grade_system)