def submit_response(self, assessment_section_id, item_id, answer_form):
        """Submits an answer to an item.

        arg:    assessment_section_id (osid.id.Id): ``Id`` of the
                ``AssessmentSection``
        arg:    item_id (osid.id.Id): ``Id`` of the ``Item``
        arg:    answer_form (osid.assessment.AnswerForm): the response
        raise:  IllegalState - ``has_assessment_section_begun()`` is
                ``false or is_assessment_section_over()`` is ``true``
        raise:  InvalidArgument - one or more of the elements in the
                form is invalid
        raise:  NotFound - ``assessment_section_id`` or ``item_id`` is
                not found, or ``item_id`` not part of
                ``assessment_section_id``
        raise:  NullArgument - ``assessment_section_id, item_id,`` or
                ``answer_form`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - ``answer_form`` is not of this service
        *compliance: mandatory -- This method must be implemented.*

        """
        if not isinstance(answer_form, ABCAnswerForm):
            raise errors.InvalidArgument('argument type is not an AnswerForm')

        # OK, so the following should actually NEVER be true. Remove it?
        if answer_form.is_for_update():
            raise errors.InvalidArgument('the AnswerForm is for update only, not submit')
        #

        try:
            if self._forms[answer_form.get_id().get_identifier()] == SUBMITTED:
                raise errors.IllegalState('answer_form already used in a submit transaction')
        except KeyError:
            raise errors.Unsupported('answer_form did not originate from this assessment session')
        if not answer_form.is_valid():
            raise errors.InvalidArgument('one or more of the form elements is invalid')
        answer_form._my_map['_id'] = ObjectId()
        self.get_assessment_section(assessment_section_id).submit_response(item_id, answer_form)
        self._forms[answer_form.get_id().get_identifier()] = SUBMITTED