def update_answer(self, answer_form):
        """Updates an existing answer.

        arg:    answer_form (osid.assessment.AnswerForm): the form
                containing the elements to be updated
        raise:  IllegalState - ``answer_form`` already used in an update
                transaction
        raise:  InvalidArgument - the form contains an invalid value
        raise:  NullArgument - ``answer_form`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        raise:  Unsupported - ``answer_form`` did not originate from
                ``get_answer_form_for_update()``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.repository.AssetAdminSession.update_asset_content_template
        from dlkit.abstract_osid.assessment.objects import AnswerForm as ABCAnswerForm
        collection = JSONClientValidated('assessment',
                                         collection='Item',
                                         runtime=self._runtime)
        if not isinstance(answer_form, ABCAnswerForm):
            raise errors.InvalidArgument('argument type is not an AnswerForm')
        if not answer_form.is_for_update():
            raise errors.InvalidArgument('the AnswerForm is for update only, not create')
        try:
            if self._forms[answer_form.get_id().get_identifier()] == UPDATED:
                raise errors.IllegalState('answer_form already used in an update transaction')
        except KeyError:
            raise errors.Unsupported('answer_form did not originate from this session')
        if not answer_form.is_valid():
            raise errors.InvalidArgument('one or more of the form elements is invalid')
        item_id = Id(answer_form._my_map['itemId']).get_identifier()
        item = collection.find_one(
            {'$and': [{'_id': ObjectId(item_id)},
                      {'assigned' + self._catalog_name + 'Ids': {'$in': [str(self._catalog_id)]}}]})
        index = 0
        found = False
        for i in item['answers']:
            if i['_id'] == ObjectId(answer_form._my_map['_id']):
                item['answers'].pop(index)
                item['answers'].insert(index, answer_form._my_map)
                found = True
                break
            index += 1
        if not found:
            raise errors.NotFound()
        try:
            collection.save(item)
        except:  # what exceptions does mongodb save raise?
            raise errors.OperationFailed()
        self._forms[answer_form.get_id().get_identifier()] = UPDATED
        # Note: this is out of spec. The OSIDs don't require an object to be returned:
        from .objects import Answer

        return Answer(
            osid_object_map=answer_form._my_map,
            runtime=self._runtime,
            proxy=self._proxy)