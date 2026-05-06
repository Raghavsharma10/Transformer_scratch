def get_composition(self):
        """Gets the Composition corresponding to this asset.

        return: (osid.repository.Composition) - the composiiton
        raise:  IllegalState - ``is_composition()`` is ``false``
        raise:  OperationFailed - unable to complete request
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.learning.Activity.get_objective
        if not bool(self._my_map['compositionId']):
            raise errors.IllegalState('composition empty')
        mgr = self._get_provider_manager('REPOSITORY')
        if not mgr.supports_composition_lookup():
            raise errors.OperationFailed('Repository does not support Composition lookup')
        lookup_session = mgr.get_composition_lookup_session(proxy=getattr(self, "_proxy", None))
        lookup_session.use_federated_repository_view()
        return lookup_session.get_composition(self.get_composition_id())