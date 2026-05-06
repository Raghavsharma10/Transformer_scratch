async def status(cls):
        '''
        Returns the current status of the configured API server.
        '''
        rqst = Request(cls.session, 'GET', '/manager/status')
        rqst.set_json({
            'status': 'running',
        })
        async with rqst.fetch() as resp:
            return await resp.json()