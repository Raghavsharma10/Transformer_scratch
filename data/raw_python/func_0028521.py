async def freeze(cls, force_kill: bool = False):
        '''
        Freezes the configured API server.
        Any API clients will no longer be able to create new compute sessions nor
        create and modify vfolders/keypairs/etc.
        This is used to enter the maintenance mode of the server for unobtrusive
        manager and/or agent upgrades.

        :param force_kill: If set ``True``, immediately shuts down all running
            compute sessions forcibly. If not set, clients who have running compute
            session are still able to interact with them though they cannot create
            new compute sessions.
        '''
        rqst = Request(cls.session, 'PUT', '/manager/status')
        rqst.set_json({
            'status': 'frozen',
            'force_kill': force_kill,
        })
        async with rqst.fetch() as resp:
            assert resp.status == 204