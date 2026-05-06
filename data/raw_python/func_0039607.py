def create(self, resource, data):
        '''
        A base function that performs a default create POST request for a given object
        '''

        service_def, resource_def, path = self._get_service_information(
            resource)
        self._validate(resource, data)

        return self.call(path=path, data=data, method='post')