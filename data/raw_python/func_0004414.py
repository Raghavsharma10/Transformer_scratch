def acl_remove_draft(self, id_vlan, type_acl):
        """
            Remove Acl draft by type

            :param id_vlan: Identity of Vlan
            :param type_acl: Acl type v4 or v6

            :return: None

            :raise VlanDoesNotExistException: Vlan Does Not Exist.
            :raise InvalidIdVlanException: Invalid id for Vlan.
            :raise NetworkAPIException: Failed to access the data source.
        """

        parameters = dict(id_vlan=id_vlan, type_acl=type_acl)

        uri = 'api/vlan/acl/remove/draft/%(id_vlan)s/%(type_acl)s/' % parameters

        return super(ApiVlan, self).get(uri)