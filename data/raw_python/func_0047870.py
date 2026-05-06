def update_repository(self, repository_form=None):
        """Updates an existing repository.

        :param repository_form: the form containing the elements to be updated
        :type repository_form: ``osid.repository.RepositoryForm``
        :raise: ``IllegalState`` -- ``repository_form`` already used in an update transaction
        :raise: ``InvalidArgument`` -- the form contains an invalid value
        :raise: ``NullArgument`` -- ``repository_form`` is ``null``
        :raise: ``OperationFailed`` -- unable to complete request
        :raise: ``PermissionDenied`` -- authorization failure
        :raise: ``Unsupported`` -- ``repository_form`` did not originate from ``get_repository_form_for_update()``

        *compliance: mandatory -- This method must be implemented.*

        """
        if repository_form is None:
            raise NullArgument()
        if not isinstance(repository_form, abc_repository_objects.RepositoryForm):
            raise InvalidArgument('argument type is not a RepositoryForm')
        if not repository_form.is_for_update():
            raise InvalidArgument('form is for create only, not update')

        # Check for "sandbox" genus type.  Hardcoded for now:
        if repository_form._my_map['genusTypeId'] != 'mc3-objectivebank%3Amc3.learning.objectivebank.sandbox%40MIT-OEIT':
            raise PermissionDenied('Handcar only supports updating \'sandbox\' type Repositories')

        try:
            if self._forms[repository_form.get_id().get_identifier()] == UPDATED:
                raise IllegalState('form already used in an update transaction')
        except KeyError:
            raise Unsupported('form did not originate from this session')
        if not repository_form.is_valid():
            raise InvalidArgument('one or more of the form elements is invalid')

        url_path = construct_url('objective_banks')
        try:
            result = self._put_request(url_path, repository_form._my_map)
        except Exception:
            raise  # OperationFailed
        self._forms[repository_form.get_id().get_identifier()] = UPDATED
        return objects.Repository(result)