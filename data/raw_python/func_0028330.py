async def check_presets(cls):
        '''
        Lists all resource presets in the current scaling group with additiona
        information.
        '''
        rqst = Request(cls.session, 'POST', '/resource/check-presets')
        async with rqst.fetch() as resp:
           return await resp.json()