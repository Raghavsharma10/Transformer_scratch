def find_destination_type(self, resource_url):
        '''
        Given a resource_url, figure out what it would resolve into
        '''
        resolvers = self.converters.values()
        for resolver in resolvers:
            # Not all resolvers are opinionated about destination types
            if not hasattr(resolver, 'get_destination_type'):
                continue

            destination_type = resolver.get_destination_type(resource_url)
            if destination_type:
                return destination_type