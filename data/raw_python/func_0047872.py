def delete_repository(self, repository_id=None):
        """Deletes a ``Repository``.

        :param repository_id: the ``Id`` of the ``Repository`` to remove
        :type repository_id: ``osid.id.Id``
        :raise: ``NotFound`` -- ``repository_id`` not found
        :raise: ``NullArgument`` -- ``repository_id`` is ``null``
        :raise: ``OperationFailed`` -- unable to complete request
        :raise: ``PermissionDenied`` -- authorization failure

        *compliance: mandatory -- This method must be implemented.*

        """
        if repository_id is None:
            raise NullArgument()
        if not isinstance(repository_id, Id):
            raise InvalidArgument('argument type is not an osid Id')

        # Check for "sandbox" genus type.  Hardcoded for now:
        try:
            repository = RepositoryLookupSession(proxy=self._proxy,
                                                 runtime=self._runtime).get_repository(repository_id)
        except Exception:
            raise
        if repository._my_map['genusTypeId'] != 'mc3-objectivebank%3Amc3.learning.objectivebank.sandbox%40MIT-OEIT':
            raise PermissionDenied('Handcar only supports deleting \'sandbox\' type Repositories')

        url_path = construct_url('objective_banks',
                                 bank_id=repository_id)
        result = self._delete_request(url_path)
        return objects.Repository(result)