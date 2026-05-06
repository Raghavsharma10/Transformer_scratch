def update_grade(self, grade_form):
        """Updates an existing grade.

        arg:    grade_form (osid.grading.GradeForm): the form containing
                the elements to be updated
        raise:  IllegalState - ``grade_form`` already used in an update
                transaction
        raise:  InvalidArgument - the form contains an invalid value
        raise:  NullArgument - ``grade_id`` or ``grade_form`` is
                ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - ``grade_form`` did not originate from
                ``get_grade_form_for_update()``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.repository.AssetAdminSession.update_asset_content_template
        from dlkit.abstract_osid.grading.objects import GradeForm as ABCGradeForm
        collection = JSONClientValidated('grading',
                                         collection='GradeSystem',
                                         runtime=self._runtime)
        if not isinstance(grade_form, ABCGradeForm):
            raise errors.InvalidArgument('argument type is not an GradeForm')
        if not grade_form.is_for_update():
            raise errors.InvalidArgument('the GradeForm is for update only, not create')
        try:
            if self._forms[grade_form.get_id().get_identifier()] == UPDATED:
                raise errors.IllegalState('grade_form already used in an update transaction')
        except KeyError:
            raise errors.Unsupported('grade_form did not originate from this session')
        if not grade_form.is_valid():
            raise errors.InvalidArgument('one or more of the form elements is invalid')
        grade_system_id = Id(grade_form._my_map['gradeSystemId']).get_identifier()
        grade_system = collection.find_one(
            {'$and': [{'_id': ObjectId(grade_system_id)},
                      {'assigned' + self._catalog_name + 'Ids': {'$in': [str(self._catalog_id)]}}]})
        index = 0
        found = False
        for i in grade_system['grades']:
            if i['_id'] == ObjectId(grade_form._my_map['_id']):
                grade_system['grades'].pop(index)
                grade_system['grades'].insert(index, grade_form._my_map)
                found = True
                break
            index += 1
        if not found:
            raise errors.NotFound()
        try:
            collection.save(grade_system)
        except:  # what exceptions does mongodb save raise?
            raise errors.OperationFailed()
        self._forms[grade_form.get_id().get_identifier()] = UPDATED
        # Note: this is out of spec. The OSIDs don't require an object to be returned:
        from .objects import Grade

        return Grade(
            osid_object_map=grade_form._my_map,
            runtime=self._runtime,
            proxy=self._proxy)