def get_item_form_for_create(self, item_record_types):
        """Gets the assessment item form for creating new assessment items.

        A new form should be requested for each create transaction.

        arg:    item_record_types (osid.type.Type[]): array of item
                record types to be included in the create operation or
                an empty list if none
        return: (osid.assessment.ItemForm) - the assessment item form
        raise:  NullArgument - ``item_record_types`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        raise:  Unsupported - unable to get form for requested record
                types
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceAdminSession.get_resource_form_for_create_template
        for arg in item_record_types:
            if not isinstance(arg, ABCType):
                raise errors.InvalidArgument('one or more argument array elements is not a valid OSID Type')
        if item_record_types == []:
            obj_form = objects.ItemForm(
                bank_id=self._catalog_id,
                runtime=self._runtime,
                effective_agent_id=self.get_effective_agent_id(),
                proxy=self._proxy)
        else:
            obj_form = objects.ItemForm(
                bank_id=self._catalog_id,
                record_types=item_record_types,
                runtime=self._runtime,
                effective_agent_id=self.get_effective_agent_id(),
                proxy=self._proxy)
        self._forms[obj_form.get_id().get_identifier()] = not CREATED
        return obj_form