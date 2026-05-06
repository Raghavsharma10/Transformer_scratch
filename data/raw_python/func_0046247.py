def get_provider_link_ids(self):
        """Gets the resource Ids representing the source of this asset in
        order from the most recent provider to the originating source.

        return: (osid.id.IdList) - the provider Ids
        compliance: mandatory - This method must be implemented.

        """
        id_list = []
        for id_ in self._my_map['providerLinkIds']:
            id_list.append(Id(id_))
        return IdList(id_list)