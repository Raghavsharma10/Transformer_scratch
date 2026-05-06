def _make_api(self, service_name):
        '''
        not yet in use ..
        '''

        resources = [resource for resource, resource_details in
                     service_definitions.get(service_name, {}).get("resources", {}).items()]

        for resource in resources:
            setattr(self, 'list_{0}' . format(resource), self.list)
            setattr(self, 'get_{0}' . format(resource), self.get)
            setattr(self, 'create_{0}' . format(resource), self.create)
            setattr(self, 'update_{0}' . format(resource), self.update)
            setattr(self, 'delete_{0}' . format(resource), self.delete)