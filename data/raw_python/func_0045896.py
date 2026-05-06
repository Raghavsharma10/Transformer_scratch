def get_provider_links(self):
        """Gets the ``Resources`` representing the source of this asset in order from the most recent provider to the originating source.

        return: (osid.resource.ResourceList) - the provider chain
        raise:  OperationFailed - unable to complete request
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.learning.Activity.get_assets_template
        if not bool(self._my_map['providerLinkIds']):
            raise errors.IllegalState('no providerLinkIds')
        mgr = self._get_provider_manager('RESOURCE')
        if not mgr.supports_resource_lookup():
            raise errors.OperationFailed('Resource does not support Resource lookup')

        # What about the Proxy?
        lookup_session = mgr.get_resource_lookup_session(proxy=getattr(self, "_proxy", None))
        lookup_session.use_federated_bin_view()
        return lookup_session.get_resources_by_ids(self.get_provider_link_ids())