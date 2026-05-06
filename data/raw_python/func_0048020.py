def create_answer(self, answer_form):
        """Creates a new ``Answer``.

        arg:    answer_form (osid.assessment.AnswerForm): the form for
                this ``Answer``
        return: (osid.assessment.Answer) - the new ``Answer``
        raise:  IllegalState - ``answer_form`` already used in a create
                transaction
        raise:  InvalidArgument - one or more of the form elements is
                invalid
        raise:  NullArgument - ``answer_form`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        raise:  Unsupported - ``answer_form`` did not originate from
                ``get_answer_form_for_create()``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.repository.AssetAdminSession.create_asset_content_template
        from dlkit.abstract_osid.assessment.objects import AnswerForm as ABCAnswerForm
        collection = JSONClientValidated('assessment',
                                         collection='Item',
                                         runtime=self._runtime)
        if not isinstance(answer_form, ABCAnswerForm):
            raise errors.InvalidArgument('argument type is not an AnswerForm')
        if answer_form.is_for_update():
            raise errors.InvalidArgument('the AnswerForm is for update only, not create')
        try:
            if self._forms[answer_form.get_id().get_identifier()] == CREATED:
                raise errors.IllegalState('answer_form already used in a create transaction')
        except KeyError:
            raise errors.Unsupported('answer_form did not originate from this session')
        if not answer_form.is_valid():
            raise errors.InvalidArgument('one or more of the form elements is invalid')
        answer_form._my_map['_id'] = ObjectId()
        item_id = Id(answer_form._my_map['itemId']).get_identifier()
        item = collection.find_one(
            {'$and': [{'_id': ObjectId(item_id)},
                      {'assigned' + self._catalog_name + 'Ids': {'$in': [str(self._catalog_id)]}}]})
        item['answers'].append(answer_form._my_map)
        result = collection.save(item)

        self._forms[answer_form.get_id().get_identifier()] = CREATED
        from .objects import Answer
        return Answer(
            osid_object_map=answer_form._my_map,
            runtime=self._runtime,
            proxy=self._proxy)