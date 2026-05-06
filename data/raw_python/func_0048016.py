def get_question_form_for_create(self, item_id, question_record_types):
        """Gets the question form for creating new questions.

        A new form should be requested for each create transaction.

        arg:    item_id (osid.id.Id): an assessment item ``Id``
        arg:    question_record_types (osid.type.Type[]): array of
                question record types to be included in the create
                operation or an empty list if none
        return: (osid.assessment.QuestionForm) - the question form
        raise:  NullArgument - ``question_record_types`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        raise:  Unsupported - unable to get form for requested record
                types
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.learning.ActivityAdminSession.get_activity_form_for_create_template

        if not isinstance(item_id, ABCId):
            raise errors.InvalidArgument('argument is not a valid OSID Id')
        for arg in question_record_types:
            if not isinstance(arg, ABCType):
                raise errors.InvalidArgument('one or more argument array elements is not a valid OSID Type')
        if question_record_types == []:
            # WHY are we passing bank_id = self._catalog_id below, seems redundant:
            obj_form = objects.QuestionForm(
                bank_id=self._catalog_id,
                item_id=item_id,
                catalog_id=self._catalog_id,
                runtime=self._runtime,
                proxy=self._proxy)
        else:
            obj_form = objects.QuestionForm(
                bank_id=self._catalog_id,
                record_types=question_record_types,
                item_id=item_id,
                catalog_id=self._catalog_id,
                runtime=self._runtime,
                proxy=self._proxy)
        obj_form._for_update = False
        self._forms[obj_form.get_id().get_identifier()] = not CREATED
        return obj_form