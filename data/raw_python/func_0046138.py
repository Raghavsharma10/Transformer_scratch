def get_branding_ids(self):
        """Gets the branding asset ``Ids``.

        return: (osid.id.IdList) - a list of asset ``Ids``
        *compliance: mandatory -- This method must be implemented.*

        """
        if 'brandingIds' not in self.my_osid_object._my_map:
            return IdList([])
        id_list = []
        for idstr in self.my_osid_object._my_map['brandingIds']:
            id_list.append(Id(idstr))
        return IdList(id_list)