def get_question_form_for_update(self, question_id):
        """Gets the question form for updating an existing question.

        A new question form should be requested for each update
        transaction.

        arg:    question_id (osid.id.Id): the ``Id`` of the ``Question``
        return: (osid.assessment.QuestionForm) - the question form
        raise:  NotFound - ``question_id`` is not found
        raise:  NullArgument - ``question_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method must be implemented.*

        """
        collection = JSONClientValidated('assessment',
                                         collection='Item',
                                         runtime=self._runtime)
        if not isinstance(question_id, ABCId):
            raise errors.InvalidArgument('the argument is not a valid OSID Id')
        document = collection.find_one({'question._id': ObjectId(question_id.get_identifier())})
        obj_form = objects.QuestionForm(osid_object_map=document['question'],
                                        runtime=self._runtime,
                                        proxy=self._proxy)
        self._forms[obj_form.get_id().get_identifier()] = not UPDATED
        return obj_form