def pool_by_environmentvip(self, environment_vip_id):
        """
        Method to return list object pool by environment vip id
        Param environment_vip_id: environment vip id
        Return list object pool
        """

        uri = 'api/v3/pool/environment-vip/%s/' % environment_vip_id

        return super(ApiPool, self).get(uri)