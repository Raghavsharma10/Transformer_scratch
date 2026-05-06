def create(self, vrfs):
        """
        Method to create vrf's

        :param vrfs: List containing vrf's desired to be created on database
        :return: None
        """

        data = {'vrfs': vrfs}
        return super(ApiVrf, self).post('api/v3/vrf/', data)