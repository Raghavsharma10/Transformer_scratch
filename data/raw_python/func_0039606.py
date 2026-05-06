def info(self, resource):
        '''
        prints information/documentation on a provided resource
        '''
        service_def, resource_def, path = self._get_service_information(
            resource)

        print (resource)
        print ("*******************************************")
        print ("Base URL: {0}" . format (self.tld))
        print ("Resource path: {0}" . format (resource_def.get("endpoint")))
        print ("Required parameters: {0}" . format (resource_def.get("required_params")))
        print ("Optional parameters" . format (resource_def.get("optional_params")))