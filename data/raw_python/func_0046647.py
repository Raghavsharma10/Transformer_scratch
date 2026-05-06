def get_authorizations_for_function(self, function_id):
        """Gets a list of ``Authorizations`` associated with a given function.

        In plenary mode, the returned list contains all known
        authorizations or an error results. Otherwise, the returned list
        may contain only those authorizations that are accessible
        through this session.

        arg:    function_id (osid.id.Id): a function ``Id``
        return: (osid.authorization.AuthorizationList) - the returned
                ``Authorization list``
        raise:  NullArgument - ``function_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.learning.ActivityLookupSession.get_activities_for_objective_template
        # NOTE: This implementation currently ignores plenary view
        collection = JSONClientValidated('authorization',
                                         collection='Authorization',
                                         runtime=self._runtime)
        result = collection.find(
            dict({'functionId': str(function_id)},
                 **self._view_filter()))
        return objects.AuthorizationList(result, runtime=self._runtime)