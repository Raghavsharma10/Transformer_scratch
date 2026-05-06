def pool_by_id(self, pool_id):
        """
        Method to return object pool by id
        Param pool_id: pool id
        Returns object pool
        """

        uri = 'api/v3/pool/%s/' % pool_id

        return super(ApiPool, self).get(uri)