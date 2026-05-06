async def list_files(self, path: Union[str, Path] = '.'):
        '''
        Gets the list of files in the given path inside the compute session
        container.

        :param path: The directory path in the compute session.
        '''
        params = {}
        if self.owner_access_key:
            params['owner_access_key'] = self.owner_access_key
        rqst = Request(self.session,
                       'GET', '/kernel/{}/files'.format(self.kernel_id),
                       params=params)
        rqst.set_json({
            'path': path,
        })
        async with rqst.fetch() as resp:
            return await resp.json()