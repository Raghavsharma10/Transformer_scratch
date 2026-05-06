def get_api_id(self):
        """
        Return the API ID.

        :return: API ID
        :rtype: str
        """
        logger.debug('Connecting to AWS apigateway API')
        conn = client('apigateway')
        apis = conn.get_rest_apis()
        api_id = None
        for api in apis['items']:
            if api['name'] == self.config.func_name:
                api_id = api['id']
                logger.debug('Found API id: %s', api_id)
                break
        if api_id is None:
            raise Exception('Unable to find ReST API named %s' %
                            self.config.func_name)
        return api_id