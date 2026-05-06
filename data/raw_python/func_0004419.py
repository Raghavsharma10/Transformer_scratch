def create(self, vlans):
        """
        Method to create vlan's

        :param vlans: List containing vlan's desired to be created on database
        :return: None
        """

        data = {'vlans': vlans}
        return super(ApiVlan, self).post('api/v3/vlan/', data)