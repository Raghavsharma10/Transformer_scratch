def update(self, vlans):
        """
        Method to update vlan's

        :param vlans: List containing vlan's desired to updated
        :return: None
        """

        data = {'vlans': vlans}
        vlans_ids = [str(vlan.get('id')) for vlan in vlans]

        return super(ApiVlan, self).put('api/v3/vlan/%s/' %
                                        ';'.join(vlans_ids), data)