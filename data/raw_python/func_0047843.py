def get_assets(self):
        """Gets all ``Assets``.

        In plenary mode, the returned list contains all known assets or
        an error results. Otherwise, the returned list may contain only
        those assets that are accessible through this session.

        :return: a list of ``Assets``
        :rtype: ``osid.repository.AssetList``
        :raise: ``OperationFailed`` -- unable to complete request
        :raise: ``PermissionDenied`` -- authorization failure

        *compliance: mandatory -- This method must be implemented.*

        """
        url_path = construct_url('assets',
                                 bank_id=self._catalog_idstr)
        return objects.AssetList(self._get_request(url_path))