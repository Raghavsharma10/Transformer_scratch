def delete_catalog(self, catalog_id):
        """Deletes a ``Catalog``.

        arg:    catalog_id (osid.id.Id): the ``Id`` of the ``Catalog``
                to remove
        raise:  NotFound - ``catalog_id`` not found
        raise:  NullArgument - ``catalog_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        if self._catalog_session is not None:
            return self._catalog_session.delete_catalog(catalog_id=bin_id)
        collection = JSONClientValidated('cataloging',
                                         collection='Catalog',
                                         runtime=self._runtime)
        if not isinstance(catalog_id, ABCId):
            raise errors.InvalidArgument('the argument is not a valid OSID Id')
        collection.delete_one({'_id': ObjectId(catalog_id.get_identifier())})