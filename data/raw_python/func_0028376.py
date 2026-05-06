async def get_info(self):
        '''
        Retrieves a brief information about the compute session.
        '''
        params = {}
        if self.owner_access_key:
            params['owner_access_key'] = self.owner_access_key
        rqst = Request(self.session,
                       'GET', '/kernel/{}'.format(self.kernel_id),
                       params=params)
        async with rqst.fetch() as resp:
            return await resp.json()