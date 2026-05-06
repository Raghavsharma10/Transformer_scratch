def create_repository(self, repository_form=None):
        """Creates a new ``Repository``.

        :param repository_form: the form for this ``Repository``
        :type repository_form: ``osid.repository.RepositoryForm``
        :return: the new ``Repository``
        :rtype: ``osid.repository.Repository``
        :raise: ``IllegalState`` -- ``repository_form`` already used in a create transaction
        :raise: ``InvalidArgument`` -- one or more of the form elements is invalid
        :raise: ``NullArgument`` -- ``repository_form`` is ``null``
        :raise: ``OperationFailed`` -- unable to complete request
        :raise: ``PermissionDenied`` -- authorization failure
        :raise: ``Unsupported`` -- ``repository_form`` did not originate from ``get_repository_form_for_create()``

        *compliance: mandatory -- This method must be implemented.*

        """
        if repository_form is None:
            raise NullArgument()
        if not isinstance(repository_form, abc_repository_objects.RepositoryForm):
            raise InvalidArgument('argument type is not a RepositoryForm')
        if repository_form.is_for_update():
            raise InvalidArgument('form is for update only, not create')
        try:
            if self._forms[repository_form.get_id().get_identifier()] == CREATED:
                raise IllegalState('form already used in a create transaction')
        except KeyError:
            raise Unsupported('form did not originate from this session')
        if not repository_form.is_valid():
            raise InvalidArgument('one or more of the form elements is invalid')

        url_path = construct_url('objective_banks')
        try:
            result = self._post_request(url_path, repository_form._my_map)
        except Exception:
            raise  # OperationFailed
        self._forms[repository_form.get_id().get_identifier()] = CREATED
        return objects.Repository(result)