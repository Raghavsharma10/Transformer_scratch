def __get_headers(self, action):
        '''
        Create HTTP headers for different HTTP requests. Content-type and
            Accept are 'application/json', as the library is interacting with
            a REST API.

        :param action: Action for which to create the header ('GET', 'PUT',
            'DELETE', 'SEARCH').
        :return: dict containing key-value pairs, e.g. 'Accept',
            'Content-Type', etc. (depening on the action).
        '''
        header = {}
        accept = 'application/json'
        content_type = 'application/json'

        if action is 'GET':
            header['Accept'] = accept


        elif action is 'PUT' or action is 'DELETE':

            if self.__authentication_method == self.__auth_methods['cert']:
                header['Authorization'] = 'Handle clientCert="true"'

            elif self.__authentication_method == self.__auth_methods['user_pw']:
                header['Authorization'] = 'Basic ' + self.__basic_authentication_string

            if action is 'PUT':
                header['Content-Type'] = content_type


        else:
            LOGGER.debug('__getHeader: ACTION is unknown ('+action+')')
        return header