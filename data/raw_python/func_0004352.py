def update(self, pools):
        """
        Method to update pool's

        :param pools: List containing pool's desired to updated
        :return: None
        """

        data = {'server_pools': pools}
        pools_ids = [str(pool.get('id'))
                     for pool in pools]

        return super(ApiPool, self).put('api/v3/pool/%s/' %
                                        ';'.join(pools_ids), data)