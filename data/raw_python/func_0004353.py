def create(self, pools):
        """
        Method to create pool's

        :param pools: List containing pool's desired to be created on database
        :return: None
        """

        data = {'server_pools': pools}
        return super(ApiPool, self).post('api/v3/pool/', data)