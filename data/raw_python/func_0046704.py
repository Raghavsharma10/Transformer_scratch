def create_grade(self, grade_form):
        """Creates a new ``Grade``.

        arg:    grade_form (osid.grading.GradeForm): the form for this
                ``Grade``
        return: (osid.grading.Grade) - the new ``Grade``
        raise:  IllegalState - ``grade_form`` already used in a create
                transaction
        raise:  InvalidArgument - one or more of the form elements is
                invalid
        raise:  NullArgument - ``grade_form`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - ``grade_form`` did not originate from
                ``get_grade_form_for_create()``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.repository.AssetAdminSession.create_asset_content_template
        from dlkit.abstract_osid.grading.objects import GradeForm as ABCGradeForm
        collection = JSONClientValidated('grading',
                                         collection='GradeSystem',
                                         runtime=self._runtime)
        if not isinstance(grade_form, ABCGradeForm):
            raise errors.InvalidArgument('argument type is not an GradeForm')
        if grade_form.is_for_update():
            raise errors.InvalidArgument('the GradeForm is for update only, not create')
        try:
            if self._forms[grade_form.get_id().get_identifier()] == CREATED:
                raise errors.IllegalState('grade_form already used in a create transaction')
        except KeyError:
            raise errors.Unsupported('grade_form did not originate from this session')
        if not grade_form.is_valid():
            raise errors.InvalidArgument('one or more of the form elements is invalid')
        grade_form._my_map['_id'] = ObjectId()
        grade_system_id = Id(grade_form._my_map['gradeSystemId']).get_identifier()
        grade_system = collection.find_one(
            {'$and': [{'_id': ObjectId(grade_system_id)},
                      {'assigned' + self._catalog_name + 'Ids': {'$in': [str(self._catalog_id)]}}]})
        grade_system['grades'].append(grade_form._my_map)
        result = collection.save(grade_system)

        self._forms[grade_form.get_id().get_identifier()] = CREATED
        from .objects import Grade
        return Grade(
            osid_object_map=grade_form._my_map,
            runtime=self._runtime,
            proxy=self._proxy)