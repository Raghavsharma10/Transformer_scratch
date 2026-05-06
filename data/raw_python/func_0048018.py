def update_question(self, question_form):
        """Updates an existing question.

        arg:    question_form (osid.assessment.QuestionForm): the form
                containing the elements to be updated
        raise:  IllegalState - ``question_form`` already used in an
                update transaction
        raise:  InvalidArgument - the form contains an invalid value
        raise:  NullArgument - ``question_form`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        raise:  Unsupported - ``question_form`` did not originate from
                ``get_question_form_for_update()``
        *compliance: mandatory -- This method must be implemented.*

        """
        collection = JSONClientValidated('assessment',
                                         collection='Item',
                                         runtime=self._runtime)
        if not isinstance(question_form, ABCQuestionForm):
            raise errors.InvalidArgument('argument type is not an QuestionForm')
        if not question_form.is_for_update():
            raise errors.InvalidArgument('the QuestionForm is for update only, not create')
        try:
            if self._forms[question_form.get_id().get_identifier()] == UPDATED:
                raise errors.IllegalState('question_form already used in an update transaction')
        except KeyError:
            raise errors.Unsupported('question_form did not originate from this session')
        if not question_form.is_valid():
            raise errors.InvalidArgument('one or more of the form elements is invalid')
        item_id = Id(question_form._my_map['itemId']).get_identifier()
        item = collection.find_one({'$and': [{'_id': ObjectId(item_id)},
                                   {'assigned' + self._catalog_name + 'Ids': {'$in': [str(self._catalog_id)]}}]})
        item['question'] = question_form._my_map
        try:
            collection.save(item)
        except:  # what exceptions does mongodb save raise?
            raise errors.OperationFailed()
        self._forms[question_form.get_id().get_identifier()] = UPDATED
        # Note: this is out of spec. The OSIDs don't require an object to be returned:
        return objects.Question(osid_object_map=question_form._my_map,
                                runtime=self._runtime,
                                proxy=self._proxy)