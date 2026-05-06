def update(self, resource, resource_id, data):
        '''
        A base function that performs a default create PATCH request for a given object
        '''
        service_def, resource_def, path = self._get_service_information(
            resource)

        update_path = "{0}{1}/" . format(path, resource_id)
        return self.call(path=update_path, data=data, method='patch')