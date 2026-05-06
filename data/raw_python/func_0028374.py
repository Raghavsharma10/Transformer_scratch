async def restart(self):
        '''
        Restarts the compute session.
        The server force-destroys the current running container(s), but keeps their
        temporary scratch directories intact.
        '''
        params = {}
        if self.owner_access_key:
            params['owner_access_key'] = self.owner_access_key
        rqst = Request(self.session,
                       'PATCH', '/kernel/{}'.format(self.kernel_id),
                       params=params)
        async with rqst.fetch():
            pass