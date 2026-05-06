def update_item(self, item_form):
        """Updates an existing item.

        arg:    item_form (osid.assessment.ItemForm): the form
                containing the elements to be updated
        raise:  IllegalState - ``item_form`` already used in an update
                transaction
        raise:  InvalidArgument - the form contains an invalid value
        raise:  NullArgument - ``item_form`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        raise:  Unsupported - ``item_form`` did not originate from
                ``get_item_form_for_update()``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceAdminSession.update_resource_template
        collection = JSONClientValidated('assessment',
                                         collection='Item',
                                         runtime=self._runtime)
        if not isinstance(item_form, ABCItemForm):
            raise errors.InvalidArgument('argument type is not an ItemForm')
        if not item_form.is_for_update():
            raise errors.InvalidArgument('the ItemForm is for update only, not create')
        try:
            if self._forms[item_form.get_id().get_identifier()] == UPDATED:
                raise errors.IllegalState('item_form already used in an update transaction')
        except KeyError:
            raise errors.Unsupported('item_form did not originate from this session')
        if not item_form.is_valid():
            raise errors.InvalidArgument('one or more of the form elements is invalid')
        collection.save(item_form._my_map)

        self._forms[item_form.get_id().get_identifier()] = UPDATED

        # Note: this is out of spec. The OSIDs don't require an object to be returned:
        return objects.Item(
            osid_object_map=item_form._my_map,
            runtime=self._runtime,
            proxy=self._proxy)