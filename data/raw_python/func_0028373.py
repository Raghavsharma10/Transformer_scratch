async def destroy(self):
        '''
        Destroys the compute session.
        Since the server literally kills the container(s), all ongoing executions are
        forcibly interrupted.
        '''
        params = {}
        if self.owner_access_key:
            params['owner_access_key'] = self.owner_access_key
        rqst = Request(self.session,
                       'DELETE', '/kernel/{}'.format(self.kernel_id),
                       params=params)
        async with rqst.fetch() as resp:
            if resp.status == 200:
                return await resp.json()