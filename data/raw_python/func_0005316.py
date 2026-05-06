def get_queryset(self):
        """
        This view should return a list of all the addresses the identity has
        for the supplied query parameters.
        Currently only supports address_type and default params
        Always excludes addresses with optedout = True
        """
        identity_id = self.kwargs["identity_id"]
        address_type = self.kwargs["address_type"]
        use_ct = "use_communicate_through" in self.request.query_params
        default_only = "default" in self.request.query_params
        if use_ct:
            identity = Identity.objects.select_related("communicate_through").get(
                id=identity_id
            )
            if identity.communicate_through is not None:
                identity = identity.communicate_through
        else:
            identity = Identity.objects.get(id=identity_id)
        addresses = identity.get_addresses_list(address_type, default_only)
        return [Address(addr) for addr in addresses]