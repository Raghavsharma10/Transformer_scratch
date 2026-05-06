def get_children(self):
        """Gets the children of this composition.

        return: (osid.repository.CompositionList) - the composition
                children
        raise:  OperationFailed - unable to complete request
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.learning.Activity.get_assets_template
        if not bool(self._my_map['childIds']):
            raise errors.IllegalState('no childIds')
        mgr = self._get_provider_manager('REPOSITORY')
        if not mgr.supports_composition_lookup():
            raise errors.OperationFailed('Repository does not support Composition lookup')

        # What about the Proxy?
        lookup_session = mgr.get_composition_lookup_session(proxy=getattr(self, "_proxy", None))
        lookup_session.use_federated_repository_view()
        return lookup_session.get_compositions_by_ids(self.get_child_ids())