def delete(self, resource, resource_id):
        '''
        A base function that performs a default delete DELETE request for a given object
        '''

        service_def, resource_def, path = self._get_service_information(
            resource)
        delete_path = "{0}{1}/" . format(path, resource_id)
        return self.call(path=delete_path, method="delete")