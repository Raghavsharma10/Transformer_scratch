async def download(self, resource_url):
        '''
        Download given Resource URL by finding path through graph and applying
        each step
        '''
        resolver_path = self.find_path_from_url(resource_url)
        await self.apply_resolver_path(resource_url, resolver_path)