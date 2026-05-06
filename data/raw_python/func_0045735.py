def get_family(self):
        """Gets the ``Family`` associated with this session.

        return: (osid.relationship.Family) - the family
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceLookupSession.get_bin
        from ..osid.osid_errors import OperationFailed, PermissionDenied
        from .objects import Family
        try:
            return Family(self.my_catalog_model)
        except:
            raise OperationFailed()